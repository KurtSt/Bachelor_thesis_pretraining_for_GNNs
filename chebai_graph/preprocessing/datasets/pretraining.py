from typing import Callable

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data.data import Data as GeomData

from chebai_graph.preprocessing.properties import MolecularProperty

HYDROGEN_MASS = Chem.GetPeriodicTable().GetAtomicWeight(1)


def one_hot_to_index(values: torch.Tensor) -> torch.Tensor:
    """Class index of each one-hot row, -1 for all-zero rows (unknown or not set)."""
    values = values.reshape(values.shape[0], -1)
    index = values.argmax(dim=1)
    index[values.sum(dim=1) == 0] = -1
    return index.long()


def _row_index(
    row: pd.Series | dict, prop: MolecularProperty, length: int
) -> torch.Tensor:
    """one_hot_to_index of a property of a row, all -1 if the property is missing."""
    values = row[prop.name]
    if isinstance(values, torch.Tensor):
        values = prop.encoder.decompress(values)
    if not isinstance(values, torch.Tensor) or values.numel() == 0:
        return torch.full((length,), -1, dtype=torch.long)
    return one_hot_to_index(values)


class PretrainingTargetsMixIn:
    """
    Adds the targets of the pretext tasks to every graph.

    The targets are read from the encoded properties before they are merged into the
    feature matrix, so they do not depend on the column layout of x:

    - y_atom_type (per node): atom type class, -1 for non-atom nodes
    - y_atom_num_hs (per node): class of the number of hydrogens, -1 for non-atom nodes
    - y_bond_type (per edge): bond type class, -1 for edges that are not chemical bonds
    - y_fg_type (per node): functional group class, -1 for non-FG nodes (augmented only)
    - y_num_atoms, y_num_bonds, y_num_hydrogens, y_mol_weight, y_total_charge
      (per graph)

    The model draws the masks for every batch. Must come before the property mixin in
    the base classes.
    """

    ATOM_TYPE = "AtomType"
    BOND_TYPE = "BondType"
    FG_TYPE = "AtomFunctionalGroup"
    CHARGE = "AtomCharge"
    NUM_HS = "AtomNumHs"

    def _merge_props_into_base(
        self, row: pd.Series | dict, *args, **kwargs
    ) -> GeomData | None:
        """Merge the properties into the graph and add the pretraining targets."""
        geom_data = super()._merge_props_into_base(row, *args, **kwargs)
        if geom_data is None:
            return None
        num_nodes = geom_data.x.shape[0]
        is_atom = getattr(geom_data, "is_atom_node", None)
        if is_atom is None:
            # Flat graph: every node is an atom
            is_atom = torch.ones(num_nodes, dtype=torch.bool)
            geom_data.is_atom_node = is_atom
        is_atom = is_atom.bool()
        src, tgt = geom_data.edge_index
        is_bond = is_atom[src] & is_atom[tgt]

        atom_type = _row_index(row, self._property(self.ATOM_TYPE), num_nodes)
        atom_type[~is_atom] = -1
        geom_data.y_atom_type = atom_type

        num_hs = _row_index(row, self._property(self.NUM_HS), num_nodes)
        num_hs[~is_atom] = -1
        geom_data.y_atom_num_hs = num_hs

        # Bond properties are stored once per bond, edge_index lists all edges in one
        # direction and then all edges in the other
        num_edges = geom_data.edge_index.shape[1]
        bond_type = _row_index(
            row, self._property(self.BOND_TYPE), num_edges // 2
        ).repeat(2)
        bond_type[~is_bond] = -1
        geom_data.y_bond_type = bond_type

        keys = row.index if isinstance(row, pd.Series) else row
        if self.FG_TYPE in keys:
            is_fg = geom_data.is_fg_node.bool()
            fg_type = _row_index(row, self._property(self.FG_TYPE), num_nodes)
            fg_type[~is_fg] = -1
            geom_data.y_fg_type = fg_type

        charges = self._atom_sum(row, self.CHARGE, num_nodes, is_atom)
        hydrogens = self._atom_sum(row, self.NUM_HS, num_nodes, is_atom)

        geom_data.y_num_atoms = is_atom.sum().reshape(1).float()
        geom_data.y_num_bonds = (is_bond.sum() / 2).reshape(1).float()
        geom_data.y_num_hydrogens = hydrogens.reshape(1).float()
        geom_data.y_mol_weight = (
            (
                self._atom_sum(
                    row, self.ATOM_TYPE, num_nodes, is_atom, self._atomic_weight
                )
                + hydrogens * HYDROGEN_MASS
            )
            .reshape(1)
            .float()
        )
        geom_data.y_total_charge = charges.reshape(1).float()
        return geom_data

    def _atom_sum(
        self,
        row: pd.Series | dict,
        name: str,
        num_nodes: int,
        is_atom: torch.Tensor,
        value_of: Callable[[str], float] = float,
    ) -> torch.Tensor:
        """
        Sum of a one-hot property with numeric tokens over the atoms of a molecule, e.g.
        the charge, the hydrogens or, with value_of, the atomic weight.
        """
        index = _row_index(row, self._property(name), num_nodes)
        values = self._token_values(name, value_of)[index.clamp(min=0)]
        values[(index < 0) | ~is_atom] = 0
        return values.sum()

    def _property(self, name: str) -> MolecularProperty:
        return next(p for p in self.properties if p.name == name)

    @staticmethod
    def _atomic_weight(token: str) -> float:
        return Chem.GetPeriodicTable().GetAtomicWeight(int(token))

    def _token_values(
        self, name: str, value_of: Callable[[str], float] = float
    ) -> torch.Tensor:
        """Numeric value of every class of a property encoder, cached."""
        cache = self.__dict__.setdefault("_token_values_cache", {})
        key = (name, value_of)
        if key not in cache:
            encoder = self._property(name).encoder
            values = torch.zeros(len(encoder.cache), dtype=torch.float64)
            for token, idx in encoder.cache.items():
                values[idx] = value_of(token)
            cache[key] = values
        return cache[key]
