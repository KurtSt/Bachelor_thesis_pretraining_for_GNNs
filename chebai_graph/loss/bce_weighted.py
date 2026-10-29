import os

import torch
from chebai.loss.bce_weighted import BCEWeighted


class BCEWeightedFromLabels(BCEWeighted):
    """
    BCEWeighted with the same weights, computed from the labels only.

    BCEWeighted of chebai loads the whole processed dataset, which for graph datasets
    merges the property features into every graph just to count labels. Here only the
    labels are read, and the counts are cached next to the processed data.
    """

    COUNTS_FILE = "label_counts.pt"

    def _label_counts(self) -> torch.Tensor:
        """Number of positive labels per class, cached in COUNTS_FILE."""
        extractor = self.data_extractor
        path = os.path.join(extractor.processed_dir, self.COUNTS_FILE)
        if os.path.isfile(path):
            return torch.load(path)
        counts = None
        for file_name in extractor.processed_file_names:
            rows = extractor.load_processed_data_from_file(file_name)
            labels = torch.stack([torch.Tensor(row["labels"]) for row in rows])
            del rows
            counts = labels.sum(dim=0) if counts is None else counts + labels.sum(dim=0)
        torch.save(counts, path)
        return counts

    def set_pos_weight(self, input: torch.Tensor) -> None:
        """
        Weights as in BCEWeighted: (1 - beta) / (1 - beta^n) for a class with n positive
        labels, normalized to a mean of 1.
        """
        if self.pos_weight is None:
            print(
                f"Computing loss-weights based on v{self.data_extractor.chebi_version} "
                f"dataset (beta={self.beta}) from label counts"
            )
            value_counts = self._label_counts()
            weights = [
                (1 - self.beta) / (1 - pow(self.beta, value)) for value in value_counts
            ]
            mean = sum(weights) / len(weights)
            self.pos_weight = torch.tensor(
                [w / mean for w in weights], device=input.device
            )
