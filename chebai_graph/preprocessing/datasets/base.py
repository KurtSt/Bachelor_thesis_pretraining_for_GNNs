import os
from abc import ABC
from collections.abc import Callable
from pprint import pformat
from typing import Optional

import pandas as pd
import torch
import tqdm
from chebai.preprocessing.datasets.base import XYBaseDataModule
from lightning_utilities.core.rank_zero import rank_zero_info
from rdkit import Chem
from torch_geometric.data.data import Data as GeomData

from chebai_graph.preprocessing.datasets.utils import resolve_property
from chebai_graph.preprocessing.properties import (
    AllNodeTypeProperty,
    AtomNodeTypeProperty,
    AtomProperty,
    BondProperty,
    FGNodeTypeProperty,
    MolecularProperty,
    MoleculeProperty,
)
from chebai_graph.preprocessing.reader import (
    GraphPropertyReader,
    RandomFeatureInitializationReader,
)
from chebai_graph.preprocessing.property_encoder import OneHotEncoder
from chebai_graph.preprocessing.reader.augmented_reader import _AugmentorReader


class DataPropertiesSetter(XYBaseDataModule, ABC):
    """Mixin for adding molecular property encodings to graph-based given datasets."""

    READER = GraphPropertyReader
    # Molecules whose (augmented) graphs are held in memory at once
    PROPERTY_CHUNK_SIZE = 10000

    def __init__(
        self,
        properties: list | None = None,
        transform: Callable | None = None,
        **kwargs,
    ):
        """
        Initialize GraphPropertiesMixIn.

        Args:
            properties: Optional list of MolecularProperty class paths or instances.
            transform: Optional transformation applied to each data sample.
        """
        super().__init__(**kwargs)
        # atom_properties and bond_properties are given as lists containing class_paths
        if properties is not None:
            properties = [resolve_property(prop, self.data_type) for prop in properties]
            properties = self._sort_properties(properties)
        else:
            properties = []
        self.properties: list[MolecularProperty] = properties
        assert isinstance(self.properties, list) and all(
            isinstance(p, MolecularProperty) for p in self.properties
        )
        self.transform = transform

    @property
    def data_type(self) -> str:
        """
        Token vocabulary of the property encoders (bin/<data_type>). chebai up to 1.4.0
        does not define data_type, then the ChEBI vocabulary is used.
        """
        try:
            return super().data_type
        except (AttributeError, NotImplementedError):
            return "chebi"

    @staticmethod
    def _decompressed(
        row: pd.Series | dict, prop: MolecularProperty
    ) -> torch.Tensor | float:
        """Value of a property in a row, expanded if it is stored compressed."""
        value = row[prop.name]
        if isinstance(value, torch.Tensor):
            return prop.encoder.decompress(value)
        return value

    def _sort_properties(
        self, properties: list[MolecularProperty]
    ) -> list[MolecularProperty]:
        return sorted(properties, key=lambda prop: self.get_property_path(prop))

    def _setup_properties(self) -> None:
        """
        Process and cache molecular properties to disk.

        Returns:
            None
        """
        os.makedirs(self.processed_properties_dir, exist_ok=True)
        missing = [
            prop
            for prop in self.properties
            if not os.path.isfile(self.get_property_path(prop))
        ]
        if not missing:
            # All caches exist, the molecules need not be loaded
            return

        raw_data = []

        try:
            file_names = self.processed_main_file_names
        except NotImplementedError:
            file_names = self.raw_file_names

        for file in file_names:
            # processed_dir_main only exists for ChEBI datasets
            path = os.path.join(
                (
                    self.processed_dir_main
                    if hasattr(self, "processed_dir_main")
                    else self.raw_dir
                ),
                file,
            )
            raw_data += list(self._load_dict(path))

        idents = [row["ident"] for row in raw_data]
        features = [row["features"] for row in raw_data]
        del raw_data

        # Work in chunks of molecules: the raw values of each chunk go to disk, then
        # every property is encoded chunk by chunk
        rank_zero_info(
            f"Reading values of properties {', '.join(p.name for p in missing)}..."
        )
        chunk_dir = os.path.join(self.processed_properties_dir, "_chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_files = []
        # Tokens in order of first occurrence, as the one-hot encoders expect
        tokens = {prop.name: {} for prop in missing}
        for start in tqdm.tqdm(range(0, len(features), self.PROPERTY_CHUNK_SIZE)):
            mols = self._graph_objects(
                features[start : start + self.PROPERTY_CHUNK_SIZE]
            )
            chunk = {}
            for prop in missing:
                values = [
                    self.reader.read_property(mol, prop) if mol is not None else None
                    for mol in mols
                ]
                if isinstance(prop.encoder, OneHotEncoder):
                    for value in values:
                        for token in value or ():
                            if token is not None:
                                tokens[prop.name].setdefault(token)
                chunk[prop.name] = values
            del mols
            path = os.path.join(chunk_dir, f"{start}.pt")
            torch.save(chunk, path)
            chunk_files.append((start, path))
            del chunk

        for prop in missing:
            rank_zero_info(f"Encoding property values of {prop.name}...")
            prop.encoder.on_start(property_values=[list(tokens.pop(prop.name))])
            encoded = []
            for start, path in tqdm.tqdm(chunk_files):
                values = torch.load(path, weights_only=False)[prop.name]
                # Compress right away, the lists of per-atom tensors are large
                encoded += [
                    {
                        prop.name: prop.encoder.compress(
                            torch.cat([prop.encoder.encode(v) for v in value])
                        ),
                        "ident": ident,
                    }
                    for value, ident in zip(values, idents[start:])
                    if value is not None and len(value) > 0
                ]
                del values
            # Temporary file first, an interrupted run must not leave a truncated cache
            path = self.get_property_path(prop)
            torch.save(encoded, path + ".part")
            os.replace(path + ".part", path)
            del encoded
            prop.on_finish()
        for _, path in chunk_files:
            os.remove(path)
        os.rmdir(chunk_dir)

    def _graph_objects(self, features: list) -> list:
        """Objects to read the properties from: augmented graphs or the molecules."""
        if not isinstance(self.reader, _AugmentorReader):
            return features
        mols = []
        for mol in features:
            try:
                # Copy, augmenting annotates the atoms and bonds in place
                result = self.reader._create_augmented_graph(Chem.Mol(mol))
            except Exception:
                result = None
            mols.append(result[1] if result is not None else None)
        return mols

    def _set_processed_data_props(self) -> None:
        """As in chebai, but keeps the loaded data for the splits, the file is read once."""
        if getattr(self, "UNLABELED", False):
            return super()._set_processed_data_props()
        filename = self.processed_file_names_dict["data"]
        data_pt = self.load_processed_data_from_file(filename)
        self._num_of_labels = len(data_pt[0]["labels"])
        self._feature_vector_size = max(len(d["features"]) for d in data_pt)
        self._reusable_data = (filename, data_pt)
        print(f"Number of samples in encoded data ({filename}): {len(data_pt)} samples")
        print(f"Number of labels for loaded data: {self._num_of_labels}")
        print(f"Feature vector size: {self._feature_vector_size}")

    def load_processed_data_from_file(self, filename: str) -> list[dict]:
        """Reuse the data loaded during setup and drop the stored augmented molecules."""
        reusable = getattr(self, "_reusable_data", None)
        if reusable is not None and reusable[0] == filename:
            # Data loaded during setup, handed over once
            self._reusable_data = None
            return reusable[1]
        data = super().load_processed_data_from_file(filename)
        for row in data:
            # Augmented readers store (graph, molecule), the molecule is only needed
            # for the property caches
            if isinstance(row.get("features"), tuple):
                row["features"] = row["features"][0]
        return data

    def _merge_properties_rowwise(
        self, base_data: list[dict], merge: Callable[[dict], GeomData | None]
    ) -> list[dict]:
        """
        Merge the cached properties into each graph, row by row. Same result as a left
        join on ident, without a pandas join per property.
        """
        lookups = {}
        for prop in self.properties:
            rank_zero_info(f"Loading property {prop.name}...")
            property_data = torch.load(self.get_property_path(prop), weights_only=False)
            # Values stay compressed until their graph is merged
            first = prop.encoder.decompress(property_data[0][prop.name])
            if len(first.shape) > 1:
                prop.encoder.set_encoding_length(first.shape[1])
            lookups[prop.name] = {str(e["ident"]): e[prop.name] for e in property_data}
            del property_data

        for row in tqdm.tqdm(base_data):
            row["ident"] = str(row["ident"])
            merged_row = dict(row)
            for name, lookup in lookups.items():
                merged_row[name] = lookup.pop(row["ident"], float("nan"))
            row["features"] = merge(merged_row)
            # apply transformation, e.g. masking for pretraining task
            if self.transform is not None:
                row["features"] = self.transform(row["features"])
        return base_data

    @property
    def processed_properties_dir(self) -> str:
        return os.path.join(self.processed_dir, "properties")

    def get_property_path(self, property: MolecularProperty) -> str:
        """
        Construct the cache path for a given molecular property.

        Args:
            property: Instance of a MolecularProperty.

        Returns:
            Path to the cached property file.
        """
        return os.path.join(
            self.processed_properties_dir,
            f"{property.name}_{property.encoder.name}.pt",
        )

    def _after_setup(self, **kwargs) -> None:
        """
        Finalize setup after ensuring properties are processed.

        Args:
            **kwargs: Additional keyword arguments passed to superclass.

        Returns:
            None
        """
        self._setup_properties()
        super()._after_setup(**kwargs)

    def _preprocess_molecule_for_pred(
        self, idx, raw_data: str | Chem.Mol, model_hparams: Optional[dict] = None
    ) -> Optional[dict]:
        """Preprocess prediction data."""
        # Add dummy labels because the collate function requires them.
        # Note: If labels are set to `None`, the collator will insert a `non_null_labels` entry into `loss_kwargs`,
        # which later causes `_get_prediction_and_labels` method in the prediction pipeline to treat the data as empty.
        result = self.reader.to_data(
            {"id": f"smiles_{idx}", "features": raw_data, "labels": [1, 2]}
        )
        # _read_data can return an updated version of the input data (e.g. augmented molecule dict) along with the GeomData object
        if isinstance(result["features"], tuple):
            result["features"], raw_data = result["features"]
        if result is None or result["features"] is None:
            return None
        for property in self.properties:
            property.encoder.eval = True
            property_value = self.reader.read_property(raw_data, property)
            if property_value is None or len(property_value) == 0:
                encoded_value = None
            else:
                encoded_value = torch.stack(
                    [property.encoder.encode(v) for v in property_value]
                )
                if len(encoded_value.shape) == 3:
                    encoded_value = encoded_value.squeeze(0)
            result[property.name] = encoded_value

        result["features"] = self._prediction_merge_props_into_base_wrapper(
            result, model_hparams
        )

        # apply transformation, e.g. masking for pretraining task
        if self.transform is not None:
            result["features"] = self.transform(result["features"])

        return result

    def _prediction_merge_props_into_base_wrapper(
        self, row: pd.Series | dict, model_hparams: Optional[dict] = None
    ) -> GeomData:
        """
        Wrapper to merge properties into base features for prediction.

        Args:
            row: A dictionary or pd.Series containing 'features' and encoded properties.
        Returns:
            A GeomData object with merged features.
        """
        return self._merge_props_into_base(row)


class GraphPropertiesMixIn(DataPropertiesSetter, ABC):
    def __init__(
        self,
        properties=None,
        transform=None,
        pad_node_features: int | None = None,
        pad_edge_features: int | None = None,
        distribution: str = "normal",
        **kwargs,
    ):
        super().__init__(properties, transform, **kwargs)
        self.pad_edge_features = int(pad_edge_features) if pad_edge_features else None
        self.pad_node_features = int(pad_node_features) if pad_node_features else None
        if self.pad_node_features or self.pad_edge_features:
            assert (
                distribution is not None
                and distribution in RandomFeatureInitializationReader.DISTRIBUTIONS
            ), (
                "When using padding for features, a valid distribution must be specified."
            )
            self.distribution = distribution
            if self.pad_node_features:
                print(
                    f"[Info] Node-level features will be padded with random"
                    f"{self.pad_node_features} values from {self.distribution} distribution."
                )
            if self.pad_edge_features:
                print(
                    f"[Info] Edge-level features will be padded with random"
                    f"{self.pad_edge_features} values from {self.distribution} distribution."
                )

        if self.properties:
            print(
                f"Data module uses these properties (ordered): {', '.join([str(p) for p in self.properties])}"
            )

    def _merge_props_into_base(self, row: pd.Series | dict) -> GeomData:
        """
        Merge encoded molecular properties into the GeomData object.

        Args:
            row: A dictionary containing 'features' and encoded properties.

        Returns:
            A GeomData object with merged features.
        """
        if isinstance(row["features"], tuple):
            geom_data, _ = row[
                "features"
            ]  # ignore additional returned data from _read_data (e.g. augmented molecule dict)
        else:
            geom_data = row["features"]
        assert isinstance(geom_data, GeomData)
        edge_attr = geom_data.edge_attr
        x = geom_data.x
        molecule_attr = torch.empty((1, 0))

        for property in self.properties:
            property_values = self._decompressed(row, property)
            if isinstance(property_values, torch.Tensor):
                if len(property_values.size()) == 0:
                    property_values = property_values.unsqueeze(0)
                if len(property_values.size()) == 1:
                    property_values = property_values.unsqueeze(1)
            else:
                property_values = torch.zeros(
                    (0, property.encoder.get_encoding_length())
                )

            if isinstance(property, AtomProperty):
                x = torch.cat([x, property_values], dim=1)
            elif isinstance(property, BondProperty):
                # Concat/Duplicate properties values for undirected graph as `edge_index` has first src to tgt edges, then tgt to src edges
                edge_attr = torch.cat(
                    [edge_attr, torch.cat([property_values, property_values], dim=0)],
                    dim=1,
                )
            elif isinstance(property, MoleculeProperty):
                molecule_attr = torch.cat([molecule_attr, property_values], dim=1)
            else:
                raise TypeError(f"Unsupported property type: {type(property).__name__}")

        if self.pad_node_features:
            padding_values = torch.empty((x.shape[0], self.pad_node_features))
            RandomFeatureInitializationReader.random_gni(
                padding_values, self.distribution
            )
            x = torch.cat([x, padding_values], dim=1)

        if self.pad_edge_features:
            padding_values = torch.empty((edge_attr.shape[0], self.pad_edge_features))
            RandomFeatureInitializationReader.random_gni(
                padding_values, self.distribution
            )
            edge_attr = torch.cat([edge_attr, padding_values], dim=1)

        return GeomData(
            x=x,
            edge_index=geom_data.edge_index,
            edge_attr=edge_attr,
        )

    def _merge_kwargs(self) -> dict:
        """Extra arguments of _merge_props_into_base for single graphs, none here."""
        return {}

    def load_processed_data(
        self, kind: Optional[str] = None, filename: Optional[str] = None
    ) -> list[dict]:
        """
        Load dataset and merge cached properties into base features.

        Args:
            filename: The path to the file to load.

        Returns:
            List of data entries, each a dictionary.
        """
        base_data = super().load_processed_data(kind, filename)
        base_data = self._merge_properties_rowwise(
            base_data, self._merge_props_into_base
        )

        prop_lengths = [
            (prop.name, prop.encoder.get_encoding_length()) for prop in self.properties
        ]

        # -------------------------- Count total node properties
        n_node_properties = sum(
            p.encoder.get_encoding_length()
            for p in self.properties
            if isinstance(p, AtomProperty)
        )

        in_channels_str = ""
        if self.pad_node_features:
            n_node_properties += self.pad_node_features
            in_channels_str += f" (with {self.pad_node_features} padded random values from {self.distribution} distribution)"

        in_channels_str = f"in_channels: {n_node_properties}" + in_channels_str

        # -------------------------- Count total edge properties
        n_edge_properties = sum(
            p.encoder.get_encoding_length()
            for p in self.properties
            if isinstance(p, BondProperty)
        )
        edge_dim_str = ""
        if self.pad_edge_features:
            n_edge_properties += self.pad_edge_features
            edge_dim_str += f" (with {self.pad_edge_features} padded random values from {self.distribution} distribution)"

        edge_dim_str = f"edge_dim: {n_edge_properties}" + edge_dim_str

        rank_zero_info(
            f"Finished loading dataset from properties.\nEncoding lengths: {prop_lengths}\n"
            f"Use following values for given parameters for model configuration: \n\t"
            f"{in_channels_str} \n\t"
            f"{edge_dim_str} \n\t"
        )

        return base_data


class GraphPropAsPerNodeType(DataPropertiesSetter, ABC):
    def __init__(self, properties=None, transform=None, **kwargs):
        super().__init__(properties, transform, **kwargs)
        # Sort properties so that AllNodeTypeProperty instances come first, rest of the properties order remain same
        first = self._sort_properties(
            [prop for prop in self.properties if isinstance(prop, AllNodeTypeProperty)]
        )
        rest = self._sort_properties(
            [
                prop
                for prop in self.properties
                if not isinstance(prop, AllNodeTypeProperty)
            ]
        )
        self.properties = first + rest
        print(
            "Properties are sorted so that `AllNodeTypeProperty` properties are first in sequence and rest of the order remains same\n",
            f"Data module uses these properties (ordered): {', '.join([str(p) for p in self.properties])}",
        )

    def load_processed_data(
        self, kind: Optional[str] = None, filename: Optional[str] = None
    ) -> list[dict]:
        """
        Load dataset and merge cached properties into base features.

        Args:
            filename: The path to the file to load.

        Returns:
            List of data entries, each a dictionary.
        """
        base_data = super().load_processed_data(kind, filename)
        n_node_properties = self._node_feature_width(log=True)
        return self._merge_properties_rowwise(
            base_data,
            lambda row: self._merge_props_into_base(
                row, max_len_node_properties=n_node_properties
            ),
        )

    def _merge_kwargs(self) -> dict:
        """Extra arguments of _merge_props_into_base for single graphs."""
        return {"max_len_node_properties": self._node_feature_width()}

    def _node_feature_width(self, log: bool = False) -> int:
        """Width of the node feature matrix (longest node type), optionally logged."""
        props_categories = {
            "AllNodeTypeProperties": [],
            "FGNodeTypeProperties": [],
            "AtomNodeTypeProperties": [],
            "GraphNodeTypeProperties": [],
            "BondProperties": [],
        }
        n_atom_node_properties, n_fg_node_properties = 0, 0
        n_bond_properties, n_graph_node_properties = 0, 0
        prop_lengths = []
        for prop in self.properties:
            prop_length = prop.encoder.get_encoding_length()
            prop_name = prop.name
            prop_lengths.append((prop_name, prop_length))
            if isinstance(prop, AllNodeTypeProperty):
                n_atom_node_properties += prop_length
                n_fg_node_properties += prop_length
                n_graph_node_properties += prop_length
                props_categories["AllNodeTypeProperties"].append(prop_name)
            elif isinstance(prop, FGNodeTypeProperty):
                n_fg_node_properties += prop_length
                props_categories["FGNodeTypeProperties"].append(prop_name)
            elif isinstance(prop, AtomNodeTypeProperty):
                n_atom_node_properties += prop_length
                props_categories["AtomNodeTypeProperties"].append(prop_name)
            elif isinstance(prop, BondProperty):
                n_bond_properties += prop_length
                props_categories["BondProperties"].append(prop_name)
            elif isinstance(prop, MoleculeProperty):
                # molecule props will be used as graph node props
                n_graph_node_properties += prop_length
                props_categories["GraphNodeTypeProperties"].append(prop_name)
            else:
                raise TypeError(f"Unsupported property type: {type(prop).__name__}")

        n_node_properties = max(
            n_atom_node_properties, n_fg_node_properties, n_graph_node_properties
        )
        if log:
            rank_zero_info(
                f"\nFinished loading dataset from properties.\nEncoding lengths: {prop_lengths}\n\n"
                f"Properties Categories:\n{pformat(props_categories)}\n\n"
                f"n_atom_node_properties: {n_atom_node_properties}, "
                f"n_fg_node_properties: {n_fg_node_properties}, "
                f"n_bond_properties: {n_bond_properties}, "
                f"n_graph_node_properties: {n_graph_node_properties}\n\n"
                f"Use following values for given parameters for model configuration: \n\t"
                f"in_channels: {n_node_properties}, edge_dim: {n_bond_properties}\n"
            )
        return n_node_properties

    def _merge_props_into_base(
        self, row: pd.Series, max_len_node_properties: int
    ) -> GeomData:
        """
        Merge encoded molecular properties into the GeomData object.

        Args:
            row: A dictionary containing 'features' and encoded properties.

        Returns:
            A GeomData object with merged features.
        """
        geom_data = row["features"]
        if geom_data is None:
            return None
        if isinstance(geom_data, tuple):
            geom_data = geom_data[
                0
            ]  # ignore additional returned data from _read_data (e.g. augmented molecule dict)
        assert isinstance(geom_data, GeomData)

        is_atom_node = geom_data.is_atom_node
        assert is_atom_node is not None, "`is_atom_node` must be set in the geom_data"
        is_graph_node = geom_data.is_graph_node
        assert is_graph_node is not None, "`is_graph_node` must be set in the geom_data"

        is_fg_node = ~is_atom_node & ~is_graph_node
        num_nodes = geom_data.x.size(0)
        edge_attr = geom_data.edge_attr
        assert edge_attr is not None, "edge_attr must be set in the geom_data"

        # Initialize node feature matrix
        assert max_len_node_properties is not None, (
            "Maximum len of node properties should not be None"
        )
        x = torch.zeros((num_nodes, max_len_node_properties))

        # Track column offsets for each node type
        atom_offset, fg_offset, graph_offset = 0, 0, 0

        for property in self.properties:
            property_values = self._decompressed(row, property).to(dtype=torch.float32)
            if isinstance(property_values, torch.Tensor):
                if len(property_values.size()) == 0:
                    property_values = property_values.unsqueeze(0)
                if len(property_values.size()) == 1:
                    property_values = property_values.unsqueeze(1)
            else:
                property_values = torch.zeros(
                    (0, property.encoder.get_encoding_length())
                )

            if isinstance(property, (AtomProperty, MoleculeProperty)):
                build_node_property_tensor_result = self._build_node_property_tensor(
                    property=property,
                    node_tensor=x,
                    atom_offset=atom_offset,
                    fg_offset=fg_offset,
                    graph_offset=graph_offset,
                    property_values=property_values,
                    is_atom_node=is_atom_node,
                    is_fg_node=is_fg_node,
                    is_graph_node=is_graph_node,
                )
                x = build_node_property_tensor_result["node_tensor"]
                atom_offset = build_node_property_tensor_result["atom_offset"]
                fg_offset = build_node_property_tensor_result["fg_offset"]
                graph_offset = build_node_property_tensor_result["graph_offset"]

            elif isinstance(property, BondProperty):
                edge_attr = self._build_edge_property_tensor(
                    edge_attr_tensor=edge_attr, property_values=property_values
                )
            else:
                raise TypeError(f"Unsupported property type: {type(property).__name__}")

            total_used_columns = max(atom_offset, fg_offset, graph_offset)
            assert total_used_columns <= max_len_node_properties, (
                f"Used {total_used_columns} columns, but max allowed is {max_len_node_properties}"
            )

        return GeomData(
            x=x,
            edge_index=geom_data.edge_index,
            edge_attr=edge_attr,
            is_atom_node=is_atom_node,
            is_fg_node=is_fg_node,
            is_graph_node=is_graph_node,
        )

    def _build_node_property_tensor(
        self,
        property: MolecularProperty,
        node_tensor: torch.Tensor,
        atom_offset: int,
        fg_offset: int,
        graph_offset: int,
        property_values: torch.Tensor,
        is_atom_node: torch.Tensor,
        is_fg_node: torch.Tensor,
        is_graph_node: torch.Tensor,
    ) -> dict:
        enc_len = property_values.shape[1]
        # -------------- Node properties ---------------
        if isinstance(property, AllNodeTypeProperty):
            node_tensor = self._fill_node_tensor_with_all_node_type_property(
                node_tensor=node_tensor,
                property_values=property_values,
                offset=atom_offset,
            )
            atom_offset += enc_len
            fg_offset += enc_len
            graph_offset += enc_len

        elif isinstance(property, AtomNodeTypeProperty):
            node_tensor = self._fill_node_tensor_with_atom_type_property(
                node_tensor=node_tensor,
                property_values=property_values,
                offset=atom_offset,
                is_atom_node=is_atom_node,
            )
            atom_offset += enc_len

        elif isinstance(property, FGNodeTypeProperty):
            node_tensor = self._fill_node_tensor_with_fg_type_property(
                node_tensor=node_tensor,
                property_values=property_values,
                offset=fg_offset,
                is_fg_node=is_fg_node,
            )
            fg_offset += enc_len

        elif isinstance(property, MoleculeProperty):
            node_tensor = self._fill_node_tensor_with_molecule_type_property(
                node_tensor=node_tensor,
                property_values=property_values,
                offset=graph_offset,
                is_graph_node=is_graph_node,
            )
            graph_offset += enc_len

        return {
            "node_tensor": node_tensor,
            "atom_offset": atom_offset,
            "fg_offset": fg_offset,
            "graph_offset": graph_offset,
        }

    def _fill_node_tensor_with_all_node_type_property(
        self, node_tensor: torch.Tensor, property_values: torch.Tensor, offset: int
    ) -> torch.Tensor:
        node_tensor[:, offset : offset + property_values.shape[1]] = property_values
        return node_tensor

    def _fill_node_tensor_with_atom_type_property(
        self,
        node_tensor: torch.Tensor,
        property_values: torch.Tensor,
        offset: int,
        is_atom_node: torch.Tensor,
    ) -> torch.Tensor:
        # We need a to mask property values
        # node_tensor.shape : torch.Size([85, 203])
        # is_atom_node.shape : torch.Size([85])
        # property_values.shape : torch.Size([85, 1])
        node_tensor[is_atom_node, offset : offset + property_values.shape[1]] = (
            property_values[is_atom_node]
        )
        return node_tensor

    def _fill_node_tensor_with_fg_type_property(
        self,
        node_tensor: torch.Tensor,
        property_values: torch.Tensor,
        offset: int,
        is_fg_node: torch.Tensor,
    ) -> torch.Tensor:
        node_tensor[is_fg_node, offset : offset + property_values.shape[1]] = (
            property_values[is_fg_node]
        )
        return node_tensor

    def _fill_node_tensor_with_molecule_type_property(
        self,
        node_tensor: torch.Tensor,
        property_values: torch.Tensor,
        offset: int,
        is_graph_node: torch.Tensor,
    ) -> torch.Tensor:
        # No mask for graph properties is required
        # is_graph_node.shape : torch.Size([85])
        # node_tensor.shape : torch.Size([85, 203])
        # property_values.shape : torch.Size([1, 200])
        node_tensor[is_graph_node, offset : offset + property_values.shape[1]] = (
            property_values
        )
        return node_tensor

    def _build_edge_property_tensor(
        self,
        edge_attr_tensor: torch.Tensor,
        property_values: torch.Tensor,
    ) -> torch.Tensor:
        # Concat/Duplicate properties values for undirected graph as `edge_index` has first src to tgt edges, then tgt to src edges
        edge_attr_tensor = torch.cat(
            [
                edge_attr_tensor,
                torch.cat([property_values, property_values], dim=0),
            ],
            dim=1,
        )
        return edge_attr_tensor

    def _prediction_merge_props_into_base_wrapper(
        self, row: pd.Series | dict, model_hparams: Optional[dict] = None
    ) -> GeomData:
        """
        Wrapper to merge properties into base features for prediction.

        Args:
            row: A dictionary or pd.Series containing 'features' and encoded properties.
        Returns:
            A GeomData object with merged features.
        """
        if (
            model_hparams is None
            or "in_channels" not in model_hparams["config"]
            or model_hparams["config"]["in_channels"] is None
        ):
            raise ValueError(
                f"model_hparams must be provided for data class: {self.__class__.__name__}"
                f" which should contain 'in_channels' key with valid value in 'config' dictionary."
            )
        max_len_node_properties = int(model_hparams["config"]["in_channels"])
        return self._merge_props_into_base(row, max_len_node_properties)
