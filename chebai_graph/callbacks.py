import os
from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer


class KeepAllLoggers(Callback):
    """
    Restore all configured loggers, e.g. CSV and W&B. The CustomTrainer of chebai keeps
    only the first logger of a list.
    """

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        configured = getattr(trainer, "init_kwargs", {}).get("logger")
        if isinstance(configured, (list, tuple)) and len(configured) > len(
            trainer.loggers
        ):
            trainer.loggers = list(configured)


class TestPredictionWriter(Callback):
    """
    Save probabilities, labels and idents of the test split to test_predictions.pt in
    the log directory of the test run, for the analyses per class and per molecule.
    Probabilities are stored as float32, labels as bool.
    """

    def __init__(self, filename: str = "test_predictions.pt") -> None:
        super().__init__()
        self.filename = filename
        self._probs, self._labels, self._idents = [], [], []

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._probs, self._labels, self._idents = [], [], []

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(outputs, dict) or outputs.get("preds") is None:
            return
        self._probs.append(outputs["preds"].detach().float().cpu())
        if outputs.get("labels") is not None:
            self._labels.append(outputs["labels"].detach().cpu().bool())
        idents = getattr(batch, "additional_fields", {}).get("idents")
        self._idents.extend(
            list(idents) if idents is not None else [None] * len(outputs["preds"])
        )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._probs:
            return
        result = {
            "probs": torch.cat(self._probs),
            "labels": torch.cat(self._labels) if self._labels else None,
            "idents": self._idents,
        }
        directory = trainer.log_dir or trainer.default_root_dir
        os.makedirs(directory, exist_ok=True)
        torch.save(result, os.path.join(directory, self.filename))
