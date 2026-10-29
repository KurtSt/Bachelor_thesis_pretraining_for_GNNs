import torch

from chebai_graph.models.architectures.gat import GATGraphPred
from chebai_graph.models.augmented import GATAMGPoolGraphPred


class PretrainedEncoderMixIn:
    """
    Load the encoder (gnn.*) from a GATGraphPretraining checkpoint; pooling and head
    start from scratch. Without a checkpoint, this is the plain model.
    """

    def __init__(self, *args, pretrained_checkpoint: str | None = None, **kwargs):
        """
        Args:
            pretrained_checkpoint (str | None): GATGraphPretraining checkpoint to load the
                encoder from.
        """
        super().__init__(*args, **kwargs)
        self.pretrained_checkpoint = pretrained_checkpoint
        if pretrained_checkpoint:
            self.load_pretrained_encoder(pretrained_checkpoint)

    def load_pretrained_encoder(self, checkpoint_path: str) -> None:
        """
        Load the encoder weights of a pretraining checkpoint into gnn.

        Raises:
            ValueError: If the checkpoint contains no encoder weights.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        prefix = "gnn."
        encoder_state = {
            k[len(prefix) :]: v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith(prefix)
        }
        if not encoder_state:
            raise ValueError(f"No encoder weights found in {checkpoint_path}")
        # Architecture and feature dimensions must match exactly
        self.gnn.load_state_dict(encoder_state, strict=True)
        print(
            f"Loaded pretrained encoder ({len(encoder_state)} tensors) from {checkpoint_path}"
        )


class GATGraphPredFromPretraining(PretrainedEncoderMixIn, GATGraphPred):
    """GATGraphPred (flat graph) with an optional pretrained encoder."""


class GATAMGPoolGraphPredFromPretraining(PretrainedEncoderMixIn, GATAMGPoolGraphPred):
    """GATAMGPoolGraphPred (augmented graph) with an optional pretrained encoder."""
