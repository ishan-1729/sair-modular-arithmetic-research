"""Source-complete HRM semantics in a Candidate-sized de-novo SAIR model."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROUTES=("full_hrm","equal_frequency","high_reset","low_reset","segment_detach_disabled","full_gradient","no_deep_supervision","fixed_depth","equal_average_act","random_act","q_disabled","q_shuffled","one_state","frozen","randomized","candidate005")
TRAINED_ROUTES=ROUTES[:13]

class RotaryPostNormBlock(nn.Module):
    def __init__(self, hidden:int, heads:int, feedforward:int)->None:
        super().__init__(); self.hidden=hidden; self.heads=heads; self.head_dim=hidden//heads
        if hidden%heads or self.head_dim%2: raise ValueError("hidden/head dimensions must support RoPE")
        self.qkv=nn.Linear(hidden,3*hidden,bias=False); self.out=nn.Linear(hidden,hidden,bias=False)
        self.up=nn.Linear(hidden,2*feedforward,bias=False); self.down=nn.Linear(feedforward,hidden,bias=False)
        self.attn_norm=nn.RMSNorm(hidden); self.mlp_norm=nn.RMSNorm(hidden)
    def _rope(self,x:Tensor)->Tensor:
        length=x.shape[-2]; half=self.head_dim//2; pos=torch.arange(length,device=x.device,dtype=x.dtype)
        inv=torch.exp(-math.log(10000.0)*torch.arange(half,device=x.device,dtype=x.dtype)/half)
        phase=pos[:,None]*inv[None,:]; cos,sin=phase.cos()[None,None],phase.sin()[None,None]
        a,b=x[...,:half],x[...,half:]; return torch.cat((a*cos-b*sin,a*sin+b*cos),-1)
    def forward(self,x:Tensor)->Tensor:
        batch,length,_=x.shape; q,k,v=self.qkv(x).chunk(3,-1)
        def shape(t): return t.view(batch,length,self.heads,self.head_dim).transpose(1,2)
        q,k,v=self._rope(shape(q)),self._rope(shape(k)),shape(v)
        attended=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(batch,length,self.hidden)
        x=self.attn_norm(x+self.out(attended)); gate,value=self.up(x).chunk(2,-1)
        return self.mlp_norm(x+self.down(F.silu(gate)*value))

@dataclass
class HRMOutput:
    logits:Tensor; quotient_logits:Tensor; q_values:Tensor; high_state:Tensor; low_state:Tensor
    segment_logits:tuple[Tensor,...]; segment_quotients:tuple[Tensor,...]; segment_q:tuple[Tensor,...]
    segments:Tensor; low_updates:int; high_updates:int; state_trace:tuple[tuple[Tensor,Tensor],...]

class SourceCompleteHRM(nn.Module):
    def __init__(self,*,hidden:int=64,heads:int=4,feedforward:int=96,high_cycles:int=2,low_steps:int=2,maximum_segments:int=4,initial_state_seed:int=68321734)->None:
        super().__init__(); self.hidden=hidden; self.high_cycles=high_cycles; self.low_steps=low_steps; self.maximum_segments=maximum_segments
        self.input=nn.Conv1d(6,hidden,1,bias=False); self.low=RotaryPostNormBlock(hidden,heads,feedforward); self.high=RotaryPostNormBlock(hidden,heads,feedforward)
        self.output_norm=nn.RMSNorm(hidden); self.local_head=nn.Linear(hidden,1,bias=False); self.quotient_head=nn.Sequential(nn.Linear(2*hidden,hidden,bias=False),nn.SiLU(),nn.Linear(hidden,3,bias=False)); self.q_head=nn.Linear(hidden,2,bias=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initial_state_seed); hi=torch.empty(1,hidden,1).normal_().clamp_(-2,2); lo=torch.empty(1,hidden,1).normal_().clamp_(-2,2)
        self.register_buffer("high_initial",hi); self.register_buffer("low_initial",lo)
    def parameter_count(self)->int:return sum(p.numel() for p in self.parameters())
    def decode(self,high:Tensor)->tuple[Tensor,Tensor,Tensor]:
        seq=self.output_norm(high.transpose(1,2)); logits=self.local_head(seq).squeeze(-1); pooled=torch.cat((seq.mean(1),seq.amax(1)),1)
        return logits,self.quotient_head(pooled),torch.sigmoid(self.q_head(seq.mean(1)))
    def _segment(self,x:Tensor,high:Tensor,low:Tensor,route:str)->tuple[Tensor,Tensor]:
        equal=route=="equal_frequency"; full_grad=route=="full_gradient"; one=route=="one_state"; total=self.high_cycles*self.low_steps
        def step(h,l,index):
            l=self.low((l+h+x).transpose(1,2)).transpose(1,2)
            update_h=equal or (index+1)%self.low_steps==0
            if update_h: h=self.high((h+l).transpose(1,2)).transpose(1,2)
            return h,l
        if one:
            for index in range(total): low=self.low((low+x).transpose(1,2)).transpose(1,2)
            return low,low
        if full_grad:
            for index in range(total): high,low=step(high,low,index)
        else:
            with torch.no_grad():
                for index in range(total-1): high,low=step(high,low,index)
            high,low=step(high,low,total-1)
        return high,low
    def run_segments(self,features:Tensor,*,route:str,training:bool)->HRMOutput:
        if route not in ROUTES[:-1] or features.ndim!=3 or features.shape[1]!=6 or not torch.isfinite(features).all(): raise ValueError("invalid HRM route/features")
        x=self.input(features); batch,width=features.shape[0],features.shape[-1]; high=self.high_initial.expand(batch,-1,width); low=self.low_initial.expand(batch,-1,width)
        logs=[]; quots=[]; qs=[]; trace=[]
        for segment in range(self.maximum_segments):
            if segment and route!="segment_detach_disabled": high,low=high.detach(),low.detach()
            if route=="high_reset": high=self.high_initial.expand_as(high)
            if route=="low_reset": low=self.low_initial.expand_as(low)
            high,low=self._segment(x,high,low,route); logits,quot,q=self.decode(high); logs.append(logits);quots.append(quot);qs.append(q);trace.append((high.detach(),low.detach()))
        chosen=self.maximum_segments
        if not training:
            if route in {"fixed_depth","q_disabled"}: chosen=self.maximum_segments
            elif route=="equal_average_act": chosen=2
            elif route=="random_act": chosen=1+int(features.shape[-1]%self.maximum_segments)
            else:
                chosen=self.maximum_segments
                for index,q in enumerate(qs,1):
                    values=q.roll(1,0) if route=="q_shuffled" else q
                    if bool((values[:,0]>values[:,1]).all()): chosen=index;break
        idx=chosen-1; high_out,low_out=trace[idx]
        return HRMOutput(logs[idx],quots[idx],qs[idx],high_out,low_out,tuple(logs),tuple(quots),tuple(qs),torch.full((batch,),chosen,dtype=torch.long,device=features.device),self.maximum_segments*self.high_cycles*self.low_steps,self.maximum_segments*self.high_cycles*(self.low_steps if route=="equal_frequency" else 1),tuple(trace))
    def forward(self,features:Tensor,*,route:str="full_hrm")->HRMOutput:
        """Deterministic target-free inference: learned Q sees hidden state only."""
        return self.run_segments(features,route=route,training=False)
    def forward_train(self,features:Tensor,*,route:str="full_hrm")->HRMOutput:return self.run_segments(features,route=route,training=True)
    def randomized_intervention(self,seed:int)->None:
        g=torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters(): p.copy_(torch.randn(p.shape,generator=g,dtype=p.dtype)*.02)
