"""Two-state, one-operator RSM adaptation initialized at the EXP-609 proposal."""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU

ROUTES = ("proposal_only", "rsm_full", "one_state", "deep_supervision", "full_gradient_no_detach", "fixed_depth", "high_reset", "low_reset", "randomized_states")
TRAINED_ROUTES = ROUTES[1:-1]

@dataclass
class RSMOutput:
    logits: Tensor
    quotient_logits: Tensor
    state: Tensor
    proposal: Tensor
    high_state: Tensor
    low_state: Tensor
    relative_residual: Tensor
    iterations: Tensor
    supervision_logits: tuple[Tensor, ...]

class SharedRSMOperator(nn.Module):
    """The sole transition operator, reused for both states and every depth."""
    def __init__(self, hidden: int, inner: int, scale: float) -> None:
        super().__init__(); self.scale = float(scale)
        self.norm = nn.GroupNorm(1, 2 * hidden + 6)
        self.inner = nn.Conv1d(2 * hidden + 6, inner, 3, padding=1)
        self.output = nn.Conv1d(inner, hidden, 1)
        nn.init.kaiming_uniform_(self.inner.weight, nonlinearity="relu")
        nn.init.zeros_(self.inner.bias); nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, state: Tensor, injection: Tensor, features: Tensor) -> Tensor:
        joined = self.norm(torch.cat((state, injection, features), dim=1))
        return state + self.scale * torch.tanh(self.output(F.silu(self.inner(joined))))

class RSMCell(nn.Module):
    def __init__(self, *, hidden: int = 92, cycles: int = 2, operator_inner: int = 64, transition_scale: float = .10, inference_high_depth: int = 2, inference_low_depth: int = 2, deep_high_depth: int = 4, deep_low_depth: int = 4, detach_probability: float = .99) -> None:
        super().__init__(); self.base = SharedDilatedConvGRU(width=256, hidden=hidden, cycles=cycles)
        self.operator = SharedRSMOperator(hidden, operator_inner, transition_scale)
        self.inference_high_depth=int(inference_high_depth); self.inference_low_depth=int(inference_low_depth)
        self.deep_high_depth=int(deep_high_depth); self.deep_low_depth=int(deep_low_depth); self.detach_probability=float(detach_probability)
        for p in self.base.parameters(): p.requires_grad_(False)

    def load_source(self, state: dict[str, Tensor]) -> None: self.base.load_state_dict(state, strict=True)
    def parameter_count(self) -> int: return sum(p.numel() for p in self.parameters())
    def trainable_parameter_count(self) -> int: return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def proposal_state(self, features: Tensor) -> Tensor:
        self.base.width=features.shape[-1]; hidden=self.base.stem(features)
        for dilation in self.base.dilation_schedule(features.shape[-1]): hidden=self.base.recurrent_update(hidden, features, dilation)
        return hidden
    def decode(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        normalized=F.silu(self.base.output_norm(hidden)); logits=self.base.local_head(normalized).squeeze(1)
        pooled=torch.cat((normalized.mean(-1), normalized.amax(-1)),1); return logits,self.base.quotient_head(pooled)

    def _run(self, features: Tensor, route: str, high_depth: int, low_depth: int, include_previous: bool) -> RSMOutput:
        if route not in ROUTES: raise ValueError(f"unknown route: {route}")
        proposal=self.proposal_state(features); high=proposal; low=proposal
        if route == "proposal_only":
            logits,quotient=self.decode(proposal); zero=torch.zeros(proposal.shape[0],device=proposal.device)
            return RSMOutput(logits,quotient,proposal,proposal,high,low,zero,zero.long(),(logits,))
        outputs=[]; calls=0
        if route == "one_state":
            for step in range(high_depth*(low_depth+1)):
                low=self.operator(low,proposal,features); calls+=1
                if step+1 < high_depth*(low_depth+1): low=low.detach()
            high=low; outputs.append(self.decode(high)[0])
        else:
            for outer in range(high_depth):
                if outer < high_depth-1 and route not in {"full_gradient_no_detach"}:
                    keep = include_previous and outer == high_depth-2
                    if not keep: high,low=high.detach(),low.detach()
                if route == "high_reset": high=proposal
                if route == "low_reset": low=proposal
                for _ in range(low_depth): low=self.operator(low, high + proposal, features); calls+=1
                high=self.operator(high,low,features); calls+=1
                outputs.append(self.decode(high)[0])
        logits,quotient=self.decode(high)
        residual=(high-proposal).flatten(1).norm(dim=1)/proposal.flatten(1).norm(dim=1).clamp_min(1e-9)
        supervision=tuple(outputs if route == "deep_supervision" else [logits])
        return RSMOutput(logits,quotient,high,proposal,high,low,residual,torch.full_like(residual,calls,dtype=torch.long),supervision)

    def forward(self, features: Tensor, *, route: str = "rsm_full", high_depth: int | None = None, low_depth: int | None = None) -> RSMOutput:
        """Target-free fixed-depth inference; no verifier, stopping, or repair surface."""
        return self._run(features,route,self.inference_high_depth if high_depth is None else int(high_depth),self.inference_low_depth if low_depth is None else int(low_depth),False)

    def forward_train(self, features: Tensor, *, route: str = "rsm_full", high_depth: int = 2, low_depth: int = 2, generator: torch.Generator | None = None) -> RSMOutput:
        high_depth=max(2,int(high_depth)); include=False
        if generator is not None: include=bool(torch.rand((),generator=generator).item() >= self.detach_probability)
        return self._run(features,route,high_depth,int(low_depth),include)

    def randomized_states(self, seed: int) -> None:
        generator=torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.operator.parameters(): p.copy_(torch.randn(p.shape,generator=generator,dtype=p.dtype)*.02)
