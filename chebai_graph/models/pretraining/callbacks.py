import os
from typing import Any

from lightning.pytorch import Callback, LightningModule, Trainer

from chebai_graph.preprocessing.datasets.streaming import stream_position


class MoleculeMilestoneCheckpoint(Callback):
    """
    Save a checkpoint when the number of training molecules passes a milestone, and
    final.ckpt at the end of training.

    With the streaming PubChem datasets every molecule is seen once, so the checkpoint
    after 10 million molecules is an encoder pretrained on the first 10 million
    molecules of the stream. A resumed run continues the stream where it stopped.
    """

    def __init__(self, milestones: list[int], dirpath: str | None = None) -> None:
        """
        Args:
            milestones (list[int]): Numbers of molecules, e.g. [1_000_000].
            dirpath (str | None): Checkpoint directory, by default checkpoints/ in the
                log directory.
        """
        super().__init__()
        self.milestones = sorted(int(m) for m in milestones)
        self.dirpath = dirpath
        self.molecules_seen = 0
        self.last_position = -1
        self._trainer = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self._trainer = trainer

    def _dir(self, trainer: Trainer) -> str:
        if self.dirpath is not None:
            return self.dirpath
        return os.path.join(trainer.log_dir or trainer.default_root_dir, "checkpoints")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        before = self.molecules_seen
        idents = batch.additional_fields["idents"]
        self.molecules_seen += len(idents)
        self.last_position = max(
            self.last_position, *(stream_position(i) for i in idents)
        )
        pl_module.log(
            "molecules_seen", float(self.molecules_seen), on_step=True, on_epoch=False
        )
        for milestone in self.milestones:
            if before < milestone <= self.molecules_seen:
                path = os.path.join(self._dir(trainer), f"molecules_{milestone}.ckpt")
                trainer.save_checkpoint(path)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # The encoder that fine-tuning starts from
        trainer.save_checkpoint(os.path.join(self._dir(trainer), "final.ckpt"))

    def state_dict(self) -> dict:
        return {
            "molecules_seen": self.molecules_seen,
            "last_position": self.last_position,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.molecules_seen = state_dict["molecules_seen"]
        self.last_position = state_dict["last_position"]
        datamodule = getattr(self._trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "resume_position"):
            # Skips the few batches that workers had prefetched but not trained on
            datamodule.resume_position = self.last_position + 1
