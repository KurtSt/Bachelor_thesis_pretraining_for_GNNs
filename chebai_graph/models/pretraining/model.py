from typing import Any

import torch
from chebai.preprocessing.structures import XYData
from torch import nn
from torch_geometric.data import Data as GraphData

from chebai_graph.models.architectures.base import GraphBaseNet
from chebai_graph.models.architectures.gat import GATGraphConvNetBase

from .masking import mask_batch
from .tasks import NEEDS_ATOM_MASK, NEEDS_BOND_MASK, NEEDS_FG_MASK, build_task


class GATGraphPretraining(GraphBaseNet):
    """
    Self-supervised pretraining of a GATv2 encoder.

    Each batch is masked (see mask_batch), the encoder embeds the masked graph and every
    task predicts what was hidden. The loss is the (weighted) sum of the task losses; only
    the encoder (gnn) is transferred to fine-tuning. With tasks [atom_type], this is
    AttrMask (Hu et al., 2020), with several tasks it is multi-task pretraining as in
    Hi-Mol (Zang et al., 2023).
    """

    def __init__(
        self,
        config: dict,
        tasks: list[str],
        mask_rate: float = 0.15,
        n_shared_node_columns: int = 0,
        hide_graph_node_features: bool = True,
        task_weighting: str = "equal",
        **kwargs,
    ):
        """
        Args:
            config (dict): Encoder configuration, the same as for fine-tuning.
            tasks (list[str]): Names of the pretext tasks (see build_task).
            mask_rate (float): Masking probability of atoms, bonds and FG nodes. Only the
                masks that the tasks need are drawn.
            n_shared_node_columns (int): Leading node columns that stay visible when a node
                is masked (3 for the node type of the augmented graph, 0 for the flat graph).
            hide_graph_node_features (bool): Hide the descriptors of the graph node.
            task_weighting (str): "equal" (sum of the losses) or "learned" (softmax weights
                as in Hi-Mol).
        """
        super().__init__(**kwargs)
        self.gnn = GATGraphConvNetBase(config=config)
        dim = int(config["out_channels"])
        self.tasks = nn.ModuleDict({name: build_task(name, dim) for name in tasks})
        required = {r for task in self.tasks.values() for r in task.requires}
        self.atom_rate = mask_rate if NEEDS_ATOM_MASK in required else 0.0
        self.bond_rate = mask_rate if NEEDS_BOND_MASK in required else 0.0
        self.fg_rate = mask_rate if NEEDS_FG_MASK in required else 0.0
        self.n_shared_node_columns = n_shared_node_columns
        self.hide_graph_node_features = hide_graph_node_features
        if task_weighting not in ("equal", "learned"):
            raise ValueError(f"Unknown task weighting: {task_weighting}")
        self.task_weighting = task_weighting
        if task_weighting == "learned":
            self.task_logits = nn.Parameter(torch.zeros(len(self.tasks)))

    def forward(self, data: dict, **kwargs) -> torch.Tensor:
        """Node embeddings of the unmasked graphs."""
        return self.gnn(data)

    def _embed(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        graph = GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return self.gnn({"features": [graph]})

    def _task_weights(self) -> torch.Tensor:
        """Weights of the task losses; learned weights have a mean of 1."""
        if self.task_weighting == "learned":
            return torch.softmax(self.task_logits, dim=0) * len(self.tasks)
        return torch.ones(len(self.tasks), device=self.device)

    def _execute(
        self,
        batch: XYData,
        batch_idx: int,
        metrics: nn.Module | None = None,
        prefix: str = "",
        log: bool = True,
        sync_dist: bool = False,
    ) -> dict[str, Any]:
        """Mask the batch, embed it and sum the weighted task losses."""
        batch = batch.to(self.device)
        graph = batch.x[0].to(self.device)
        generator = None
        if prefix != "train_":
            # Fixed masks for validation and test, the losses stay comparable
            generator = torch.Generator(device=self.device)
            generator.manual_seed(batch_idx)
        masked = mask_batch(
            graph,
            atom_rate=self.atom_rate,
            bond_rate=self.bond_rate,
            fg_rate=self.fg_rate,
            n_shared_node_columns=self.n_shared_node_columns,
            hide_graph_node_features=self.hide_graph_node_features,
            generator=generator,
        )
        emb = self._embed(masked.x, masked.edge_index, masked.edge_attr)

        weights = self._task_weights()
        loss = emb.new_zeros(())
        stats: dict[str, torch.Tensor] = {}
        for weight, (name, task) in zip(weights, self.tasks.items()):
            result = task(emb, graph, masked)
            if result is None:
                continue
            task_loss, task_stats = result
            loss = loss + weight * task_loss
            stats[f"loss_{name}"] = task_loss.detach()
            stats.update({k: v.detach() for k, v in task_stats.items()})
        if not stats:
            # No targets in this batch (e.g. no FG node masked), zero loss with a grad_fn
            loss = emb.sum() * 0.0

        if log:
            num_graphs = int(graph.batch.max()) + 1
            self.log(
                f"{prefix}loss",
                loss.detach(),
                batch_size=num_graphs,
                on_step=prefix == "train_",
                on_epoch=True,
                prog_bar=True,
                sync_dist=sync_dist,
            )
            for key, value in stats.items():
                self.log(
                    f"{prefix}{key}",
                    value,
                    batch_size=num_graphs,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=sync_dist,
                )
        return dict(loss=loss, output=emb)
