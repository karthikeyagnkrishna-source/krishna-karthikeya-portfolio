"""Physical units and input-only contracts shared by training and evaluation."""
from dataclasses import dataclass
from typing import Any
import torch

@dataclass(frozen=True)
class ToleranceConfig:
    relative: float = .02
    zero_threshold: float = 1e-8
    angle: float = .05
    speed: float = .01
    torque: float = .05
    force: float = 1.
    contact: float = .02

@dataclass
class InputBatch:
    base: dict[str, torch.Tensor]
    history: torch.Tensor
    history_valid: torch.Tensor
    commands: torch.Tensor
    command_valid: torch.Tensor
    def to(self, device):
        return InputBatch({k:v.to(device) for k,v in self.base.items()}, self.history.to(device),
                          self.history_valid.to(device),self.commands.to(device),self.command_valid.to(device))

@dataclass
class Targets:
    leg: torch.Tensor
    force: torch.Tensor
    body: torch.Tensor
    contact: torch.Tensor
    times: torch.Tensor
    rows: Any
    def to(self, device):
        return Targets(*(getattr(self,k).to(device) for k in ('leg','force','body','contact','times')),self.rows)

@dataclass
class Predictions:
    leg: torch.Tensor
    force: torch.Tensor
    body: torch.Tensor
    contact: torch.Tensor
    r: torch.Tensor
    hidden: torch.Tensor | None = None
    fv: torch.Tensor | None = None

ScaleBundle = dict
