import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PromptTokenSelector(nn.Module):
    def __init__(self, n_prefix_tokens: int, n_vocab: int):
        super().__init__()
        self.n_prefix_tokens = n_prefix_tokens
        self.n_vocab = n_vocab
        self.selector = nn.Parameter(torch.randn(n_prefix_tokens, n_vocab), requires_grad=True)

    def reset_selector_probs(self, selector_probs: torch.Tensor):
        assert selector_probs.shape == (self.n_prefix_tokens, self.n_vocab)
        self.selector = nn.Parameter(selector_probs.detach().clone(), requires_grad=True)

    def soft_projection(self, projection: torch.Tensor):
        assert projection.shape == (self.n_prefix_tokens, self.n_vocab)
        self.selector = nn.Parameter(
            torch.where(projection > 1e-6, torch.maximum(projection, self.selector), self.selector)
            .detach()
            .clone(),
            requires_grad=True,
        )

    def forward(self, hard: bool = False):
        if hard:
            selected = F.gumbel_softmax(torch.log_softmax(self.selector, dim=-1), tau=1, hard=hard)
            return selected
        else:
            return F.softmax(self.selector, dim=-1)


class PromptTokenSelectorWithFreqBias(PromptTokenSelector):
    _validity_mask: torch.Tensor
    _tok_freqs: torch.Tensor

    def __init__(
        self,
        n_prefix_tokens: int,
        n_vocab: int,
        validity_mask: Optional[torch.Tensor] = None,
        token_freqs: Optional[torch.Tensor] = None,
        delta: float = 2.0,
    ):
        super().__init__(n_prefix_tokens, n_vocab)
        self.delta = delta
        if validity_mask is not None:
            self.register_buffer("_validity_mask", validity_mask)
        else:
            print("No validity mask provided. Assuming all tokens are valid.")
            self.register_buffer("_validity_mask", torch.ones(n_vocab))

        if token_freqs is not None:
            self.register_buffer("_tok_freqs", token_freqs)
        else:
            print("No token frequencies provided. Assuming all tokens are equally frequent.")
            self.register_buffer("_tok_freqs", torch.zeros(n_vocab))

        assert torch.all(token_freqs >= 0), "Token frequencies must be non-negative."
        assert torch.all(token_freqs <= 1), "Token frequencies must be less than or equal to 1."

    def reset_selector_probs(self, selector_probs: torch.Tensor):
        assert selector_probs.shape == (self.n_prefix_tokens, self.n_vocab)
        self.selector = nn.Parameter(selector_probs.detach().clone(), requires_grad=True)

    def forward(self, hard: bool = False):
        # validity mask is 1 if the token is a valid identifier
        # invert and convert to boolean mask to be used in torch.masked_fill
        mask = self._validity_mask.reshape(1, -1) == 0
        bias = self._tok_freqs.reshape(1, -1)

        # mask out invalid tokens
        selector_values = self.selector.masked_fill(mask, float("-inf"))

        # bias frequent tokens
        selector_values = selector_values + self.delta * bias

        if hard:
            selected = F.gumbel_softmax(
                torch.log_softmax(selector_values, dim=-1), tau=1, hard=hard
            )
            return selected
        else:
            return F.softmax(selector_values, dim=-1)
