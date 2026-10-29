from dataclasses import dataclass

import torch
from torch_geometric.data import Data as GraphData


@dataclass
class MaskedBatch:
    """
    Masked encoder input of a batch and what was hidden.

    Attributes:
        x (torch.Tensor): Node features; masked nodes keep only their node type columns.
        edge_index (torch.Tensor): Edges without the removed bonds.
        edge_attr (torch.Tensor): Edge features without the removed bonds.
        masked_atoms (torch.Tensor): Indices of the masked atoms, shape (K,).
        masked_fgs (torch.Tensor): Indices of the masked FG nodes, shape (K,).
        masked_bonds (torch.Tensor): Atom pairs of the removed bonds, shape (2, K).
        masked_bond_types (torch.Tensor): Types of the removed bonds, shape (K,).
        negative_pairs (torch.Tensor): Unbonded atom pairs of the same molecule, shape (2, K').
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    masked_atoms: torch.Tensor
    masked_fgs: torch.Tensor
    masked_bonds: torch.Tensor
    masked_bond_types: torch.Tensor
    negative_pairs: torch.Tensor


def _select(
    candidates: torch.Tensor, rate: float, generator: torch.Generator | None
) -> torch.Tensor:
    """Keep each candidate with probability rate."""
    if rate <= 0 or candidates.numel() == 0:
        return candidates[:0]
    keep = (
        torch.rand(candidates.numel(), device=candidates.device, generator=generator)
        < rate
    )
    return candidates[keep]


def _graph_offsets(
    graph: GraphData, is_atom: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """First node index and number of atoms of every graph in the batch."""
    num_graphs = int(graph.batch.max()) + 1
    nodes_per_graph = torch.bincount(graph.batch, minlength=num_graphs)
    first_node = torch.cumsum(nodes_per_graph, 0) - nodes_per_graph
    atoms_per_graph = torch.bincount(graph.batch[is_atom], minlength=num_graphs)
    return first_node, atoms_per_graph


def mask_batch(
    graph: GraphData,
    atom_rate: float = 0.0,
    bond_rate: float = 0.0,
    fg_rate: float = 0.0,
    n_shared_node_columns: int = 0,
    hide_graph_node_features: bool = False,
    generator: torch.Generator | None = None,
) -> MaskedBatch:
    """
    Mask a batch of graphs for pretraining.

    - Masked atoms and FG nodes keep their first n_shared_node_columns features (the
      node type on the augmented graph), all other features are set to zero.
    - Masked bonds are removed in both directions and become the targets of the bond
      tasks.
    - For every removed bond, one atom pair of the same molecule without a bond is drawn
      as negative example.
    - With hide_graph_node_features, the graph node keeps only its node type. Its RDKit
      descriptors contain targets such as the number of atoms.

    The atom nodes of a graph come first, as in the graph readers.

    Args:
        graph (GraphData): Batch with the pretraining targets.
        atom_rate (float): Masking probability of an atom.
        bond_rate (float): Removal probability of a bond.
        fg_rate (float): Masking probability of an FG node.
        n_shared_node_columns (int): Leading feature columns that stay visible.
        hide_graph_node_features (bool): Hide the descriptors of the graph node.
        generator (torch.Generator | None): Random generator for fixed masks.

    Returns:
        MaskedBatch: Masked input and the hidden entities.
    """
    x = graph.x.float().clone()
    edge_index = graph.edge_index.long()
    edge_attr = graph.edge_attr
    device = x.device
    num_nodes = x.shape[0]
    is_atom = graph.is_atom_node.bool()

    atom_candidates = torch.nonzero(graph.y_atom_type.long() >= 0).flatten()
    masked_atoms = _select(atom_candidates, atom_rate, generator)
    x[masked_atoms, n_shared_node_columns:] = 0

    if fg_rate > 0 and hasattr(graph, "y_fg_type"):
        fg_candidates = torch.nonzero(graph.y_fg_type.long() >= 0).flatten()
        masked_fgs = _select(fg_candidates, fg_rate, generator)
        x[masked_fgs, n_shared_node_columns:] = 0
    else:
        masked_fgs = torch.zeros(0, dtype=torch.long, device=device)

    if hide_graph_node_features and hasattr(graph, "is_graph_node"):
        x[graph.is_graph_node.bool(), n_shared_node_columns:] = 0

    empty_pairs = torch.zeros((2, 0), dtype=torch.long, device=device)
    masked_bonds, negative_pairs = empty_pairs, empty_pairs
    masked_bond_types = torch.zeros(0, dtype=torch.long, device=device)
    if bond_rate > 0:
        bond_type = graph.y_bond_type.long()
        is_bond = bond_type >= 0
        u = torch.minimum(edge_index[0], edge_index[1])
        v = torch.maximum(edge_index[0], edge_index[1])
        key = u * num_nodes + v
        bond_keys, inverse = torch.unique(key[is_bond], return_inverse=True)
        types = torch.zeros_like(bond_keys).scatter_(0, inverse, bond_type[is_bond])
        selected = _select(
            torch.arange(bond_keys.numel(), device=device), bond_rate, generator
        )
        masked_keys = bond_keys[selected]
        masked_bonds = torch.stack([masked_keys // num_nodes, masked_keys % num_nodes])
        masked_bond_types = types[selected]

        keep_edge = ~torch.isin(key, masked_keys)
        edge_index = edge_index[:, keep_edge]
        edge_attr = edge_attr[keep_edge]

        # Negative pairs: a random atom of the same molecule as partner
        if masked_keys.numel() > 0:
            first_node, atoms_per_graph = _graph_offsets(graph, is_atom)
            anchor = masked_bonds[0]
            g = graph.batch[anchor]
            offset = torch.floor(
                torch.rand(anchor.numel(), device=device, generator=generator)
                * atoms_per_graph[g]
            ).long()
            partner = first_node[g] + offset
            pu = torch.minimum(anchor, partner)
            pv = torch.maximum(anchor, partner)
            valid = (pu != pv) & ~torch.isin(pu * num_nodes + pv, bond_keys)
            negative_pairs = torch.stack([pu[valid], pv[valid]])

    return MaskedBatch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        masked_atoms=masked_atoms,
        masked_fgs=masked_fgs,
        masked_bonds=masked_bonds,
        masked_bond_types=masked_bond_types,
        negative_pairs=negative_pairs,
    )
