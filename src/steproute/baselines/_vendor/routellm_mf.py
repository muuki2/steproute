# Adapted from RouteLLM, routellm/routers/matrix_factorization/train_matrix_factorization.py and
# routellm/routers/matrix_factorization/model.py, at commit
# 0b64fdafe049e596a3f5657c219329f24af24198 (https://github.com/lm-sys/RouteLLM).
#
# RouteLLM is distributed under the Apache License, Version 2.0 (its LICENSE file names no
# copyright holder beyond the licence template).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The full licence text is LICENSE-RouteLLM in this directory.
#
# Modified for steproute (issue 22). Changes from the upstream files:
# - The prompt table ``Q`` holds fixed numeric routing-state vectors passed in by the caller,
#   not OpenAI ``text-embedding-3-small`` embeddings, and inference takes such a vector too: no
#   network call and no OpenAI client anywhere.
# - The model table ``P`` is sized for the caller's models; the fixed ``MODEL_IDS`` table of
#   Chatbot Arena models is gone.
# - The training loop draws batches and noise from a seeded ``torch.Generator``, runs on the
#   CPU, has no progress bar, and returns the per-epoch training loss.
# - Training and inference share one class (upstream has ``MFModel_Train`` and ``MFModel``).
"""RouteLLM's matrix-factorisation router (arXiv 2406.18665), adapted to take numeric states."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

__all__ = ["MFModel", "train_loops"]


class MFModel(nn.Module):
    """``logit(model, prompt) = w . (normalize(P[model]) * proj(Q[prompt]))``; a battle is
    scored by ``logit(winner) - logit(loser)``."""

    def __init__(
        self,
        dim: int,
        num_models: int,
        prompts: Any,
        num_classes: int = 1,
        use_proj: bool = True,
    ) -> None:
        super().__init__()
        q = torch.as_tensor(prompts, dtype=torch.float)
        text_dim = int(q.shape[1])
        self.use_proj = use_proj
        self.P = nn.Embedding(num_models, dim)
        self.Q = nn.Embedding(int(q.shape[0]), text_dim).requires_grad_(False)
        self.Q.weight.data.copy_(q)
        if self.use_proj:
            self.text_proj = nn.Linear(text_dim, dim, bias=False)
        else:
            assert text_dim == dim, f"text_dim {text_dim} must be equal to dim {dim}"
        self.classifier = nn.Linear(dim, num_classes, bias=False)  # bias must be False

    def _project(self, prompt_embed: torch.Tensor) -> torch.Tensor:
        return self.text_proj(prompt_embed) if self.use_proj else prompt_embed

    def forward(
        self,
        model_win: torch.Tensor,
        model_loss: torch.Tensor,
        prompt: torch.Tensor,
        test: bool = False,
        alpha: float = 0.05,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        model_win_embed = F.normalize(self.P(model_win), p=2, dim=1)
        model_loss_embed = F.normalize(self.P(model_loss), p=2, dim=1)
        prompt_embed = self.Q(prompt)
        if not test:
            # adding noise to stabilise the training (upstream)
            noise = torch.randn(prompt_embed.shape, generator=generator)
            prompt_embed = prompt_embed + noise * alpha
        prompt_embed = self._project(prompt_embed)
        out: torch.Tensor = self.classifier(
            (model_win_embed - model_loss_embed) * prompt_embed
        ).squeeze()
        return out

    def pred_win_rate(self, model_a: int, model_b: int, prompt_vec: Any) -> float:
        """Probability that ``model_a`` beats ``model_b`` on a new prompt vector."""
        with torch.no_grad():
            x = torch.as_tensor(prompt_vec, dtype=torch.float).reshape(1, -1)
            embed = F.normalize(self.P(torch.tensor([model_a, model_b])), p=2, dim=1)
            logits = self.classifier(embed * self._project(x)).squeeze()
            return float(torch.sigmoid(logits[0] - logits[1]))


def train_loops(
    net: MFModel,
    battles: Sequence[tuple[int, int, int]],
    *,
    lr: float,
    weight_decay: float,
    alpha: float,
    num_epochs: int,
    batch_size: int,
    generator: torch.Generator,
) -> list[float]:
    """Upstream's loop: Adam, BCE-with-logits against 1 for ``(winner, loser, prompt)``."""
    win = torch.tensor([b[0] for b in battles], dtype=torch.long)
    lose = torch.tensor([b[1] for b in battles], dtype=torch.long)
    prompt = torch.tensor([b[2] for b in battles], dtype=torch.long)
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    losses: list[float] = []
    n = len(battles)
    for _ in range(num_epochs):
        net.train()
        order = torch.randperm(n, generator=generator)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            output = net(win[idx], lose[idx], prompt[idx], alpha=alpha, generator=generator)
            output = output.reshape(-1)
            ls = loss_fn(output, torch.ones_like(output))
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
            total += float(ls.detach()) * len(idx)
        losses.append(total / n)
    net.eval()
    return losses
