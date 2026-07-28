"""Exact-initialized H2L3 hierarchy with separable MagicNorm and credit horizon."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor,nn
from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU

ROUTES=("proposal_only","full","hierarchy_only","credit_only","flat_compute_matched","no_warmup_raw_gradient","high_reset","low_reset","randomized")
TRAINED_ROUTES=ROUTES[1:-1]

def parameterless_rms_norm(x:Tensor,eps:float=1e-6)->Tensor:
    return x*torch.rsqrt(x.square().mean(dim=1,keepdim=True)+eps)

class MagicNormModule(nn.Module):
    """PreNorm residual block capped by parameterless RMSNorm."""
    def __init__(self,hidden:int,inner:int,scale:float)->None:
        super().__init__(); self.scale=float(scale); self.norm=nn.GroupNorm(1,2*hidden+6); self.up=nn.Conv1d(2*hidden+6,inner,3,padding=1); self.down=nn.Conv1d(inner,hidden,1); nn.init.kaiming_uniform_(self.up.weight,nonlinearity="relu"); nn.init.zeros_(self.up.bias); nn.init.zeros_(self.down.weight); nn.init.zeros_(self.down.bias)
    def forward(self,state:Tensor,injection:Tensor,features:Tensor,*,magic:bool)->Tensor:
        joined=torch.cat((state,injection,features),1); residual=joined+self.scale*torch.cat((self.down(F.silu(self.up(self.norm(joined)))),torch.zeros_like(joined[:,state.shape[1]:])),1); bounded=parameterless_rms_norm(residual) if magic else residual; return state+self.scale*self.down(F.silu(self.up(self.norm(bounded))))

@dataclass
class HRMOutput:
    logits:Tensor; quotient_logits:Tensor; state:Tensor; proposal:Tensor; high_state:Tensor; low_state:Tensor; relative_residual:Tensor; iterations:Tensor; supervision_logits:tuple[Tensor,...]; state_trace:tuple[Tensor,...]; credit_horizon:int

class H2L3Cell(nn.Module):
    def __init__(self,*,hidden:int=92,cycles:int=2,inner:int=64,transition_scale:float=.10,high_cycles:int=2,low_cycles:int=3,warmup_start_horizon:int=2,warmup_end_horizon:int=5)->None:
        super().__init__(); self.base=SharedDilatedConvGRU(width=256,hidden=hidden,cycles=cycles); self.high_module=MagicNormModule(hidden,inner,transition_scale); self.low_module=MagicNormModule(hidden,inner,transition_scale); self.high_cycles=int(high_cycles); self.low_cycles=int(low_cycles); self.warmup_start_horizon=int(warmup_start_horizon); self.warmup_end_horizon=int(warmup_end_horizon); self.register_buffer("low_initial",torch.zeros(1,hidden,1),persistent=True)
        for p in self.base.parameters(): p.requires_grad_(False)
    def load_source(self,state:dict[str,Tensor])->None:self.base.load_state_dict(state,strict=True)
    def parameter_count(self)->int:return sum(p.numel() for p in self.parameters())
    def trainable_parameter_count(self)->int:return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def proposal_state(self,features:Tensor)->Tensor:
        self.base.width=features.shape[-1]; hidden=self.base.stem(features)
        for dilation in self.base.dilation_schedule(features.shape[-1]):hidden=self.base.recurrent_update(hidden,features,dilation)
        return hidden
    def decode(self,state:Tensor):
        normalized=F.silu(self.base.output_norm(state)); logits=self.base.local_head(normalized).squeeze(1); pooled=torch.cat((normalized.mean(-1),normalized.amax(-1)),1); return logits,self.base.quotient_head(pooled)
    def horizon(self,progress:float)->int:return round(self.warmup_start_horizon+(self.warmup_end_horizon-self.warmup_start_horizon)*max(0.,min(1.,float(progress))))
    def _run(self,features:Tensor,route:str,credit_horizon:int)->HRMOutput:
        if route not in ROUTES:raise ValueError(route)
        proposal=self.proposal_state(features); high=proposal; low=self.low_initial.expand(proposal.shape[0],-1,proposal.shape[-1]); trace=[]
        if route=="proposal_only":
            logits,q=self.decode(proposal); zero=torch.zeros(proposal.shape[0],device=proposal.device); return HRMOutput(logits,q,proposal,proposal,high,low,zero,zero.long(),(logits,),(high,low),0)
        hierarchical=route not in {"credit_only","flat_compute_matched"}; magic=route in {"full","credit_only","no_warmup_raw_gradient","high_reset","low_reset","randomized"}; raw=route in {"hierarchy_only","flat_compute_matched","no_warmup_raw_gradient"}; total=self.high_cycles*(self.low_cycles+1); horizon=total if raw else max(1,min(total,int(credit_horizon))); step=0
        if hierarchical:
            for outer in range(self.high_cycles):
                if route=="high_reset":high=proposal
                if route=="low_reset":low=self.low_initial.expand_as(low)
                for _ in range(self.low_cycles):
                    if step==total-horizon:high,low=high.detach(),low.detach()
                    low=self.low_module(low,high+proposal,features,magic=magic); trace.append(low); step+=1
                if step==total-horizon:high,low=high.detach(),low.detach()
                high=self.high_module(high,low,features,magic=magic); trace.append(high); step+=1
        else:
            low=proposal
            for step in range(total):
                if step==total-horizon:low=low.detach()
                low=self.low_module(low,proposal,features,magic=magic); trace.append(low)
            high=low
        logits,q=self.decode(high); residual=(high-proposal).flatten(1).norm(dim=1)/proposal.flatten(1).norm(dim=1).clamp_min(1e-9); return HRMOutput(logits,q,high,proposal,high,low,residual,torch.full_like(residual,total,dtype=torch.long),(logits,),tuple(trace),horizon)
    def forward(self,features:Tensor,*,route:str="full",high_depth:int|None=None,low_depth:int|None=None)->HRMOutput:
        """Fixed H2L3, target-free inference without stopping, filtering, or repair."""
        if high_depth is not None or low_depth is not None: raise ValueError("H2L3 inference schedule is frozen")
        return self._run(features,route,self.warmup_end_horizon)
    def forward_train(self,features:Tensor,*,route:str="full",progress:float=0.)->HRMOutput:return self._run(features,route,self.horizon(progress))
    def randomized_intervention(self,seed:int)->None:
        g=torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for module in (self.high_module,self.low_module):
                for p in module.parameters():p.copy_(torch.randn(p.shape,generator=g,dtype=p.dtype)*.02)
