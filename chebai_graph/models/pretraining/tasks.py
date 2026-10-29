"""
Pretext tasks for self-supervised pretraining on flat and augmented molecular graphs.

Atom and FG level (masked inputs, Hu et al. 2020 / Zang et al. 2023):
    atom_type   masked atoms -> atom type
    atom_num_hs masked atoms -> number of attached hydrogens
    bond_type   removed bonds -> bond type
    bond_link   removed bonds vs. unbonded atom pairs -> bond exists
    fg_type     masked FG nodes -> functional group (augmented graph only)
Molecule level (read from the graph node, Zang et al. 2023):
    atom_count, bond_count, hydrogen_count, mol_weight, total_charge
"""

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data as GraphData
from torch_scatter import scatter_add

from chebai_graph.preprocessing.properties import (
    AtomFunctionalGroup,
    AtomNumHs,
    AtomType,
    BondType,
)

from .masking import MaskedBatch

# Masks a task can request from mask_batch
NEEDS_ATOM_MASK = "atom"
NEEDS_BOND_MASK = "bond"
NEEDS_FG_MASK = "fg"


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Two-layer prediction head, as in Hi-Mol."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
    )


def _pair_repr(emb: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Order-independent representation of node pairs, (2, K) -> (K, 2D)."""
    a, b = emb[pairs[0]], emb[pairs[1]]
    return torch.cat([a + b, a * b], dim=1)


def _vocabulary_size(prop_class: type) -> int:
    """Number of classes of a property in the ChEBI vocabulary."""
    return prop_class(data_type="chebi").encoder.get_encoding_length()


class PretrainingTask(nn.Module, ABC):
    """Base class of the pretext tasks. `requires` lists the masks a task needs."""

    requires: tuple[str, ...] = ()

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(
        self, emb: torch.Tensor, graph: GraphData, masked: MaskedBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
        """
        Compute the loss of the task.

        Args:
            emb (torch.Tensor): Node embeddings of the masked batch.
            graph (GraphData): Unmasked batch with the targets.
            masked (MaskedBatch): Masked input and the hidden entities.

        Returns:
            tuple | None: Loss and statistics for logging, or None if the batch has
                nothing to predict.
        """


class _ClassificationTask(PretrainingTask):
    """Cross-entropy on masked nodes or removed bonds."""

    def _classify(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
        # -1: value not in the ChEBI vocabulary, e.g. an atom with more hydrogens
        # than any ChEBI atom
        known = target >= 0
        logits, target = logits[known], target[known]
        if target.numel() == 0:
            return None
        loss = F.cross_entropy(logits, target)
        acc = (logits.argmax(dim=1) == target).float().mean()
        return loss, {f"acc_{self.name}": acc}


class AtomTypeTask(_ClassificationTask):
    """Atom type of masked atoms (AttrMask)."""

    requires = (NEEDS_ATOM_MASK,)

    def __init__(self, dim: int, name: str = "atom_type") -> None:
        super().__init__(name)
        self.head = _mlp(dim, dim, _vocabulary_size(AtomType))

    def forward(
        self, emb: torch.Tensor, graph: GraphData, masked: MaskedBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
        idx = masked.masked_atoms
        return self._classify(self.head(emb[idx]), graph.y_atom_type[idx].long())


class AtomNumHsTask(_ClassificationTask):
    """Number of hydrogens of masked atoms."""

    requires = (NEEDS_ATOM_MASK,)

    def __init__(self, dim: int, name: str = "atom_num_hs") -> None:
        super().__init__(name)
        self.head = _mlp(dim, dim, _vocabulary_size(AtomNumHs))

    def forward(
        self, emb: torch.Tensor, graph: GraphData, masked: MaskedBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
        idx = masked.masked_atoms
        return self._classify(self.head(emb[idx]), graph.y_atom_num_hs[idx].long())


class FGTypeTask(_ClassificationTask):
    """Functional group of masked FG nodes (augmented graph only)."""

    requires = (NEEDS_FG_MASK,)

    def __init__(self, dim: int, name: str = "fg_type") -> None:
        super().__init__(name)
        self.head = _mlp(dim, dim, _vocabulary_size(AtomFunctionalGroup))

    def forward(
        self, emb: torch.Tensor, graph: GraphData, masked: MaskedBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
        idx = masked.masked_fgs
        return self._classify(self.head(emb[idx]), graph.y_fg_type[idx].long())


class BondTypeTask(_ClassificationTask):
    """Type of removed bonds, predicted from both atoms."""

    requires = (NEEDS_BOND_MASK,)

    def __init__(self, dim: int, name: str = "bond_type") -> None:
        super().__init__(name)
        self.head = _mlp(2 * dim, dim, _vocabulary_size(BondType))

    def forward(
        self, emb: torch.Tensor, graph: GraphData, masked: MaskedBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
        logits = self.head(_pair_repr(emb, masked.masked_bonds))
        return self._classify(logits, masked.masked_bond_types)


class BondLinkTask(PretrainingTask):
    """Removed bonds against unbonded atom pairs of the same molecule."""

    requires = (NEEDS_BOND_MASK,)

    def __init__(self, dim: int, name: str = "bond_link") -> None:
        super().__init__(name)
        self.head = _mlp(2 * dim, dim, 1)

    def forward(
        self, emb: torch.Tensor, graph: GraphData, masked: MaskedBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
        pos, neg = masked.masked_bonds, masked.negative_pairs
        if pos.shape[1] == 0:
            return None
        pairs = torch.cat([pos, neg], dim=1)
        target = torch.cat([emb.new_ones(pos.shape[1]), emb.new_zeros(neg.shape[1])])
        logits = self.head(_pair_repr(emb, pairs)).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        acc = ((logits > 0).float() == target).float().mean()
        return loss, {f"acc_{self.name}": acc}


def graph_readout(emb: torch.Tensor, graph: GraphData) -> torch.Tensor:
    """Embedding of the graph node if there is one, otherwise the sum over all nodes."""
    num_graphs = int(graph.batch.max()) + 1
    is_graph_node = getattr(graph, "is_graph_node", None)
    if is_graph_node is not None and bool(is_graph_node.any()):
        is_graph_node = is_graph_node.bool()
        return scatter_add(
            emb[is_graph_node], graph.batch[is_graph_node], dim=0, dim_size=num_graphs
        )
    return scatter_add(emb, graph.batch, dim=0, dim_size=num_graphs)


class GraphRegressionTask(PretrainingTask):
    """
    Regression of a per-graph quantity from the graph embedding.

    Counts are regressed as log(1 + count), raw counts give losses of 10 and more.
    The logged MAE is in the original unit.
    """

    def __init__(
        self, dim: int, name: str, target: str, log_scale: bool = True
    ) -> None:
        """
        Args:
            target (str): Attribute of the batch with the target, e.g. y_num_atoms.
            log_scale (bool): Regress log(1 + target) instead of the target.
        """
        super().__init__(name)
        self.target = target
        self.log_scale = log_scale
        self.head = _mlp(dim, dim, 1)

    def forward(
        self, emb: torch.Tensor, graph: GraphData, masked: MaskedBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred = self.head(graph_readout(emb, graph)).squeeze(-1)
        target = getattr(graph, self.target).float()
        if self.log_scale:
            loss = F.smooth_l1_loss(pred, torch.log1p(target))
            value = torch.expm1(pred)
        else:
            loss = F.smooth_l1_loss(pred, target)
            value = pred
        mae = (value - target).abs().mean()
        return loss, {f"mae_{self.name}": mae}


def build_task(name: str, dim: int) -> PretrainingTask:
    """
    Create a pretext task by name.

    Args:
        name (str): One of atom_type, atom_num_hs, bond_type, bond_link, fg_type,
            atom_count, bond_count, hydrogen_count, mol_weight, total_charge.
        dim (int): Dimension of the node embeddings.

    Raises:
        ValueError: If the name is unknown.
    """
    if name == "atom_type":
        return AtomTypeTask(dim)
    if name == "atom_num_hs":
        return AtomNumHsTask(dim)
    if name == "bond_type":
        return BondTypeTask(dim)
    if name == "bond_link":
        return BondLinkTask(dim)
    if name == "fg_type":
        return FGTypeTask(dim)
    if name == "atom_count":
        return GraphRegressionTask(dim, name, "y_num_atoms")
    if name == "bond_count":
        return GraphRegressionTask(dim, name, "y_num_bonds")
    if name == "hydrogen_count":
        return GraphRegressionTask(dim, name, "y_num_hydrogens")
    if name == "mol_weight":
        return GraphRegressionTask(dim, name, "y_mol_weight")
    if name == "total_charge":
        return GraphRegressionTask(dim, name, "y_total_charge", log_scale=False)
    raise ValueError(f"Unknown pretraining task: {name}")
