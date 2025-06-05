import torch
import torch.nn as nn
from typing import Optional


class PEZSoftPrompt(nn.Module):
    def __init__(
        self, n_prefix_tokens: int, n_embed: int, init_embeds: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.n_prefix_tokens = n_prefix_tokens
        self.n_embed = n_embed

        if init_embeds is None:
            init_embeds = torch.randn(n_prefix_tokens, n_embed)
        else:
            assert init_embeds.shape == (n_prefix_tokens, n_embed)

        self.prompts = nn.Parameter(init_embeds.detach().clone(), requires_grad=True)

    def forward(self):
        return self.prompts
