import torch
import torch.nn as nn

from typing import Tuple, Union, Optional
from torchtyping import TensorType


class BaseEnvironment(nn.Module):
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        text_embed_shape: Union[int, Tuple[int, ...]]
    ):
        super().__init__()
        self.state_shape = state_shape
        self.text_embed_shape = text_embed_shape
        self.register_buffer('dummy', torch.zeros(0), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def init(self) -> Tuple[str, torch.Tensor]: # (instruction, initial state)
        raise NotImplementedError

    def forward(
        self,
        actions: torch.Tensor
    ) -> Tuple[
        TensorType[(), float],     # reward
        torch.Tensor,              # next state
        TensorType[(), bool]       # done
    ]:
        raise NotImplementedError


class MockEnvironment(BaseEnvironment):
    def init(self) -> Tuple[
        Optional[str],
        TensorType[float]
    ]:
        return 'please clean the kitchen', torch.randn(self.state_shape, device = self.device)

    def forward(self, actions) -> Tuple[
        TensorType[(), float],
        TensorType[float],
        TensorType[(), bool]
    ]:
        rewards = torch.randn((), device = self.device)
        next_states = torch.randn(self.state_shape, device = self.device)
        done = torch.zeros((), device = self.device, dtype = torch.bool)

        return rewards, next_states, done