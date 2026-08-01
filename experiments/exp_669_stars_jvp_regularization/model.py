"""Training-only one-step power-iteration JVP proxy over the unchanged EXP609 cell."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU

ROUTES=("frozen_source","stars","lambda_zero","state_norm","random_stop_gradient","randomized")
TRAINED_ROUTES=("stars","lambda_zero","state_norm","random_stop_gradient")

@dataclass
class ProxyRecord:
    proxy:Tensor; first_jvp_norm:Tensor; vector_norm:Tensor; state_norm:Tensor; dilation:int; jvp_calls:int; direction:str; parameter_gradient_enabled:bool

class STARSCell(SharedDilatedConvGRU):
    """The inference class is exactly the source class plus training-only helpers."""
    def load_source(self,state:dict[str,Tensor])->None:self.load_state_dict(state,strict=True)
    def forward_with_boundary(self,features:Tensor):
        self.width=features.shape[-1];hidden=self.stem(features);schedule=self.dilation_schedule(features.shape[-1]);boundary=None
        for index,dilation in enumerate(schedule):
            if index+1==len(schedule):boundary=hidden
            hidden=self.recurrent_update(hidden,features,dilation)
        logits,quotient=self.forward_from_hidden(hidden);assert boundary is not None;return logits,quotient,boundary,schedule[-1],hidden
    def forward_from_hidden(self,hidden:Tensor):
        import torch.nn.functional as F
        normalized=F.silu(self.output_norm(hidden));logits=self.local_head(normalized).squeeze(1);pooled=torch.cat((normalized.mean(-1),normalized.amax(-1)),1);return logits,self.quotient_head(pooled)
    def jvp_proxy(self,state:Tensor,features:Tensor,dilation:int,*,seed:int,route:str)->ProxyRecord:
        if route not in {"stars","lambda_zero","state_norm","random_stop_gradient"}:raise ValueError(route)
        generator=torch.Generator(device=state.device).manual_seed(int(seed));v0=torch.randn(state.shape,generator=generator,device=state.device,dtype=state.dtype);v0=v0/v0.flatten(1).norm(dim=1).view(-1,1,1).clamp_min(1e-12)
        transition=lambda h:self.recurrent_update(h,features,dilation)
        _,first=torch.func.jvp(transition,(state,),(v0,));v1=(first/first.flatten(1).norm(dim=1).view(-1,1,1).clamp_min(1e-12)).detach()
        direction=v0 if route=="random_stop_gradient" else v1
        _,product=torch.func.jvp(transition,(state,),(direction,));per_sample=product.flatten(1).square().sum(1);proxy=per_sample.mean()
        if route in {"lambda_zero","state_norm","random_stop_gradient"}:proxy=proxy.detach()
        return ProxyRecord(proxy,first.flatten(1).norm(dim=1).mean(),direction.flatten(1).norm(dim=1).mean(),state.flatten(1).norm(dim=1).mean(),int(dilation),2,"random" if route=="random_stop_gradient" else "one-step-power",route=="stars")
    def randomized_intervention(self,seed:int)->None:
        g=torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():p.copy_(torch.randn(p.shape,generator=g,dtype=p.dtype)*.02)
