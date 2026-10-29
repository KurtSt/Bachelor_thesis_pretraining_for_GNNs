import torch
from torch_geometric.data import Data as GeomData

from chebai_graph.preprocessing.collate import GraphCollator


def chain_graph(n_atoms: int, n_features: int = 6, augmented: bool = False) -> GeomData:
    """
    Chain molecule with pretraining targets. The augmented variant adds one FG node
    connected to all atoms and a graph node connected to the FG node.
    """
    bonds = [(i, i + 1) for i in range(n_atoms - 1)]
    n_nodes = n_atoms + (2 if augmented else 0)
    edges = list(bonds)
    if augmented:
        fg, gn = n_atoms, n_atoms + 1
        edges += [(a, fg) for a in range(n_atoms)] + [(fg, gn)]
    src = [u for u, _ in edges]
    tgt = [v for _, v in edges]
    edge_index = torch.tensor([src + tgt, tgt + src])
    is_atom = torch.zeros(n_nodes, dtype=torch.bool)
    is_atom[:n_atoms] = True
    is_bond = is_atom[edge_index[0]] & is_atom[edge_index[1]]
    data = GeomData(
        x=torch.rand((n_nodes, n_features)) + 1,
        edge_index=edge_index,
        edge_attr=torch.rand((edge_index.shape[1], 3)),
        is_atom_node=is_atom,
        y_atom_type=torch.where(is_atom, torch.arange(n_nodes) % 5, -1),
        y_atom_num_hs=torch.where(is_atom, torch.arange(n_nodes) % 3, -1),
        y_bond_type=torch.where(is_bond, 1, -1),
        y_num_atoms=torch.tensor([float(n_atoms)]),
        y_num_bonds=torch.tensor([float(len(bonds))]),
        y_num_hydrogens=torch.tensor([float(2 * n_atoms + 2)]),
        y_mol_weight=torch.tensor([14.0 * n_atoms + 2.0]),
        y_total_charge=torch.tensor([0.0]),
    )
    if augmented:
        is_graph = torch.zeros(n_nodes, dtype=torch.bool)
        is_graph[-1] = True
        is_fg = ~is_atom & ~is_graph
        data.is_graph_node = is_graph
        data.is_fg_node = is_fg
        data.y_fg_type = torch.where(is_fg, 2, -1)
    return data


def collate(graphs: list[GeomData]):
    """Collate graphs into a batch, as the data modules do."""
    rows = [
        {"features": g, "labels": None, "ident": str(i)} for i, g in enumerate(graphs)
    ]
    return GraphCollator()(rows)
