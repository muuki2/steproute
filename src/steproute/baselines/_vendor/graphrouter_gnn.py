# Adapted from LLMRouter, llmrouter/models/graphrouter/graph_nn.py, at commit
# c65ee0122bc9e84f71d571fcdec4a948481b6e88 (https://github.com/ulab-uiuc/LLMRouter).
#
# MIT License
#
# Copyright (c) 2024 U Lab @UIUC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""GraphRouter's GNN (arXiv 2410.03834) as LLMRouter implements it, adapted.

Changes from the upstream file, none of which alters the model or the training step:

- ``EncoderDecoderNet.forward`` is split into ``encode`` (node states after the two graph
  convolutions) and the edge scoring, so a new query can be scored against stored LLM node
  states (``GNNPredictor.score_queries``) without rebuilding the graph for each query. In eval
  mode this gives the same scores as the upstream ``predict`` on a graph with the query added:
  a query node has no visible edges, and batch norm uses its running statistics.
- ``FormData.formulation`` became ``build_graph``; it keeps the upstream node and edge layout.
- ``train_validate`` keeps the best state in memory instead of writing a file (starting from
  minus infinity rather than -1, so a cost-weighted performance below -1 is still kept), takes a
  seeded ``torch.Generator`` for the random edge masks, and prints nothing.
- Type annotations; the code is formatted for this repository.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.nn import GeneralConv

__all__ = ["EncoderDecoderNet", "FeatureAlign", "GNNPredictor", "build_graph"]


class FeatureAlign(nn.Module):
    """Map query and LLM features to the same dimension."""

    def __init__(self, query_feature_dim: int, llm_feature_dim: int, common_dim: int) -> None:
        super().__init__()
        self.query_transform = nn.Linear(query_feature_dim, common_dim)
        self.llm_transform = nn.Linear(llm_feature_dim, common_dim)

    def forward(self, query_features: torch.Tensor, llm_features: torch.Tensor) -> torch.Tensor:
        aligned_query_features = self.query_transform(query_features)
        aligned_llm_features = self.llm_transform(llm_features)
        return torch.cat([aligned_query_features, aligned_llm_features], 0)


class EncoderDecoderNet(nn.Module):
    """Two GeneralConv layers over query->LLM edges carrying the performance; an edge's score is
    the sigmoid of the mean of the query's aligned features times the LLM's node state."""

    def __init__(
        self, query_feature_dim: int, llm_feature_dim: int, hidden_features: int, in_edges: int = 1
    ) -> None:
        super().__init__()
        self.in_edges = in_edges
        self.model_align = FeatureAlign(query_feature_dim, llm_feature_dim, hidden_features)
        self.encoder_conv_1 = GeneralConv(
            in_channels=hidden_features, out_channels=hidden_features, in_edge_channels=in_edges
        )
        self.encoder_conv_2 = GeneralConv(
            in_channels=hidden_features, out_channels=hidden_features, in_edge_channels=in_edges
        )
        self.edge_mlp = nn.Linear(in_edges, in_edges)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.bn2 = nn.BatchNorm1d(hidden_features)

    def encode(
        self,
        query_features: torch.Tensor,
        llm_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_can_see: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(x_ini, x)``: aligned input features and node states, for every node."""
        edge_index_mask = edge_index[:, edge_can_see]
        edge_weight_mask = edge_weight[edge_can_see]
        edge_weight_mask = F.leaky_relu(self.edge_mlp(edge_weight_mask.reshape(-1, self.in_edges)))
        edge_weight_mask = edge_weight_mask.reshape(-1, self.in_edges)
        x_ini = self.model_align(query_features, llm_features)
        x = F.leaky_relu(
            self.bn1(self.encoder_conv_1(x_ini, edge_index_mask, edge_attr=edge_weight_mask))
        )
        x = self.bn2(self.encoder_conv_2(x, edge_index_mask, edge_attr=edge_weight_mask))
        return x_ini, x

    def forward(
        self,
        query_features: torch.Tensor,
        llm_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        edge_can_see: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        x_ini, x = self.encode(query_features, llm_features, edge_index, edge_can_see, edge_weight)
        edge_index_predict = edge_index[:, edge_mask]
        return torch.sigmoid((x_ini[edge_index_predict[0]] * x[edge_index_predict[1]]).mean(dim=-1))


def build_graph(
    query_feature: Any,
    llm_feature: Any,
    performance: Any,
    label: Any,
    train_mask: torch.Tensor,
    valide_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    device: str = "cpu",
) -> Data:
    """Every query joined to every LLM, query-major (edge ``q * n_llm + j``), as upstream."""
    query_features = torch.tensor(query_feature, dtype=torch.float).to(device)
    llm_features = torch.tensor(llm_feature, dtype=torch.float).to(device)
    n_q, n_llm = len(query_features), len(llm_features)
    org_node = [q for q in range(n_q) for _ in range(n_llm)]
    des_node = [n_q + j for _ in range(n_q) for j in range(n_llm)]
    edge_index = torch.tensor([org_node, des_node], dtype=torch.long).to(device)
    edge_weight = torch.tensor(performance, dtype=torch.float).reshape(-1, 1).to(device)
    return Data(
        query_features=query_features,
        llm_features=llm_features,
        edge_index=edge_index,
        edge_attr=edge_weight,
        label=torch.tensor(label, dtype=torch.float).reshape(-1, 1).to(device),
        edge_mask=edge_mask,
        train_mask=train_mask,
        valide_mask=valide_mask,
    )


class GNNPredictor:
    """Training, validation and prediction around ``EncoderDecoderNet``."""

    def __init__(
        self,
        query_feature_dim: int,
        llm_feature_dim: int,
        hidden_features_size: int,
        in_edges_size: int,
        config: dict[str, Any],
        device: str = "cpu",
    ) -> None:
        self.model = EncoderDecoderNet(
            query_feature_dim=query_feature_dim,
            llm_feature_dim=llm_feature_dim,
            hidden_features=hidden_features_size,
            in_edges=in_edges_size,
        ).to(device)
        self.config = config
        self.device = device
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        self.criterion = nn.BCELoss()
        self.num_llms = int(config.get("llm_num", 10))
        self.history: list[float] = []

    def train_validate(self, data: Data, data_validate: Data, generator: torch.Generator) -> float:
        """Upstream training loop; returns the best validation value and keeps that state."""
        best_result = -float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        train_mask = data.train_mask.clone().to(torch.bool)
        train_epochs = int(self.config.get("train_epoch", 100))
        batch_size = int(self.config.get("batch_size", 4))
        train_mask_rate = float(self.config.get("train_mask_rate", 0.3))

        for _ in range(train_epochs):
            self.model.train()
            loss_mean = torch.zeros(())
            mask_train = data.edge_mask
            for _ in range(batch_size):
                mask = mask_train.clone().bool()
                random_mask = torch.rand(mask.size(), generator=generator) < train_mask_rate
                mask = torch.where(mask & random_mask, torch.tensor(False), mask).bool()
                edge_can_see = torch.logical_and(~mask, train_mask)
                self.optimizer.zero_grad()
                predicted_edges = self.model(
                    query_features=data.query_features,
                    llm_features=data.llm_features,
                    edge_index=data.edge_index,
                    edge_mask=mask,
                    edge_can_see=edge_can_see,
                    edge_weight=data.edge_attr,
                )
                loss = self.criterion(predicted_edges.reshape(-1), data.label[mask].reshape(-1))
                loss_mean = loss_mean + loss
            loss_mean = loss_mean / batch_size
            loss_mean.backward()  # type: ignore[no-untyped-call,unused-ignore]
            self.optimizer.step()

            self.model.eval()
            mask_validate = data_validate.edge_mask.clone().to(torch.bool)
            with torch.no_grad():
                predicted_edges_validate = self.model(
                    query_features=data_validate.query_features,
                    llm_features=data_validate.llm_features,
                    edge_index=data_validate.edge_index,
                    edge_mask=mask_validate,
                    edge_can_see=train_mask,
                    edge_weight=data_validate.edge_attr,
                )
                observe_edge = predicted_edges_validate.reshape(-1, self.num_llms)
                value_validate = data_validate.edge_attr[mask_validate].reshape(-1, self.num_llms)
                row_indices = torch.arange(len(value_validate))
                max_idx = torch.argmax(observe_edge, 1).cpu()
                result_validate = float(value_validate[row_indices, max_idx].mean())
            self.history.append(result_validate)
            if result_validate > best_result:
                best_result = result_validate
                best_state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_state)
        return best_result

    def predict(self, data: Data) -> torch.Tensor:
        """Upstream prediction: the best LLM index for each query whose edges are masked."""
        self.model.eval()
        mask = data.edge_mask.clone().to(torch.bool)
        edge_can_see = torch.logical_or(data.train_mask, data.valide_mask).to(torch.bool)
        with torch.no_grad():
            edge_predict = self.model(
                query_features=data.query_features,
                llm_features=data.llm_features,
                edge_index=data.edge_index,
                edge_mask=mask,
                edge_can_see=edge_can_see,
                edge_weight=data.edge_attr,
            )
        return torch.argmax(edge_predict.reshape(-1, self.num_llms), 1)

    def llm_states(self, data: Data) -> torch.Tensor:
        """LLM node states with every training edge visible (what ``predict`` scores against)."""
        self.model.eval()
        edge_can_see = torch.logical_or(data.train_mask, data.valide_mask).to(torch.bool)
        with torch.no_grad():
            _, x = self.model.encode(
                data.query_features,
                data.llm_features,
                data.edge_index,
                edge_can_see,
                data.edge_attr,
            )
        return x[len(data.query_features) :]

    def score_queries(self, query_features: Any, llm_states: torch.Tensor) -> torch.Tensor:
        """Edge scores of new queries against stored LLM node states, one row per query."""
        self.model.eval()
        q = torch.tensor(query_features, dtype=torch.float).to(self.device)
        with torch.no_grad():
            aligned = self.model.model_align.query_transform(q)
            return torch.sigmoid((aligned[:, None, :] * llm_states[None, :, :]).mean(dim=-1))
