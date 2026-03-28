"""
CSLR models: Continuous Sign Language Recognition (§8.2, §7.2).

CSLRModel: standard single-stream conv + BiLSTM + CTC.
DualStreamCSLRModel: dual-stream (pose + motion) with feature fusion.
"""

import torch
import torch.nn as nn

from .temporal_conv import TemporalConvEncoder, MultiScaleTemporalConv
from .bilstm import BiLSTMEncoder
from .transformer import TransformerSequenceEncoder
from .heads import CTCHead


def suppress_ctc_special_tokens(
    log_probs: torch.Tensor,
    disallowed_ids: set[int] | list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Renormalize log-probs after suppressing invalid non-blank specials."""
    if not disallowed_ids:
        return log_probs

    masked = log_probs.clone()
    masked[..., list(disallowed_ids)] = torch.finfo(masked.dtype).min
    return torch.log_softmax(masked, dim=-1)


class CSLRModel(nn.Module):
    """Continuous Sign Language Recognition model with CTC head.

    Architecture: Temporal Conv → Sequence Encoder → CTC Head

    Args:
        input_dim: Per-frame feature dimension (104 or 208).
        num_classes: Vocabulary size including CTC blank.
        conv_dim: Temporal conv output channels.
        conv_layers: Number of temporal conv blocks.
        conv_kernel_size: Conv kernel size.
        conv_dropout: Dropout in conv blocks.
        encoder_type: 'bilstm' or 'transformer'.
        lstm_hidden_size: BiLSTM hidden size per direction.
        lstm_layers: Number of BiLSTM layers.
        lstm_dropout: BiLSTM/Transformer dropout.
        transformer_hidden: Transformer hidden dim.
        transformer_layers: Number of Transformer layers.
        transformer_heads: Number of attention heads.
        multi_scale: Use multi-scale temporal conv.
        multi_scale_kernels: Kernel sizes for multi-scale branches.
    """

    def __init__(
        self,
        input_dim: int = 104,
        num_classes: int = 2000,
        conv_dim: int = 384,
        conv_layers: int = 3,
        conv_kernel_size: int = 5,
        conv_dropout: float = 0.1,
        encoder_type: str = "bilstm",
        lstm_hidden_size: int = 384,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        transformer_hidden: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        multi_scale: bool = False,
        multi_scale_kernels: list[int] | None = None,
    ):
        super().__init__()

        # Temporal conv encoder
        if multi_scale:
            self.conv_encoder = MultiScaleTemporalConv(
                input_dim=input_dim,
                conv_dim=conv_dim,
                kernel_sizes=multi_scale_kernels,
                dropout=conv_dropout,
                num_layers=conv_layers,
            )
        else:
            self.conv_encoder = TemporalConvEncoder(
                input_dim=input_dim,
                conv_dim=conv_dim,
                num_layers=conv_layers,
                kernel_size=conv_kernel_size,
                dropout=conv_dropout,
            )

        # Sequence encoder
        self.encoder_type = encoder_type
        if encoder_type == "bilstm":
            self.seq_encoder = BiLSTMEncoder(
                input_dim=conv_dim,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_layers,
                dropout=lstm_dropout,
            )
            head_input_dim = self.seq_encoder.output_dim
        elif encoder_type == "transformer":
            self.seq_encoder = TransformerSequenceEncoder(
                input_dim=conv_dim,
                hidden_dim=transformer_hidden,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                dropout=lstm_dropout,
            )
            head_input_dim = self.seq_encoder.output_dim
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # CTC head
        self.ctc_head = CTCHead(
            input_dim=head_input_dim,
            num_classes=num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, input_dim) skeleton sequences.
            lengths: (B,) original sequence lengths.

        Returns:
            (B, T, num_classes) log-probabilities for CTC.
        """
        x = self.conv_encoder(x)          # (B, T, conv_dim)
        x = self.seq_encoder(x, lengths)  # (B, T, encoder_dim)
        log_probs = self.ctc_head(x)      # (B, T, num_classes)
        return log_probs

    def load_backbone(self, backbone_state_dict: dict, strict: bool = False):
        """Load pretrained backbone weights from ISLR model.

        Args:
            backbone_state_dict: State dict from ISLRModel.get_backbone_state_dict().
            strict: Whether to require exact key matching.
        """
        target_state = self.state_dict()
        filtered_state = {}
        loaded_ctc_head = False
        for name, tensor in backbone_state_dict.items():
            if not (
                name.startswith("conv_encoder.")
                or name.startswith("seq_encoder.")
            ):
                if name == "head.fc.weight" and "ctc_head.fc.weight" in target_state:
                    if target_state["ctc_head.fc.weight"].shape == tensor.shape:
                        filtered_state["ctc_head.fc.weight"] = tensor
                        loaded_ctc_head = True
                elif name == "head.fc.bias" and "ctc_head.fc.bias" in target_state:
                    if target_state["ctc_head.fc.bias"].shape == tensor.shape:
                        filtered_state["ctc_head.fc.bias"] = tensor
                        loaded_ctc_head = True
                continue
            if name not in target_state:
                continue
            if target_state[name].shape != tensor.shape:
                continue
            filtered_state[name] = tensor

        missing, unexpected = self.load_state_dict(
            filtered_state,
            strict=strict,
        )
        self._loaded_ctc_head_from_backbone = loaded_ctc_head
        if missing:
            print(f"Backbone loading — missing keys: {len(missing)}")
        if unexpected:
            print(f"Backbone loading — unexpected keys: {len(unexpected)}")
        return missing, unexpected

    def greedy_decode(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor | None = None,
        ignore_ids: set[int] | None = None,
    ) -> list[list[int]]:
        """Greedy CTC decoding (collapse repeated + remove blanks).

        Args:
            log_probs: (B, T, num_classes) log-probabilities.

        Returns:
            List of decoded gloss ID sequences, one per batch item.
        """
        predictions = log_probs.argmax(dim=-1)  # (B, T)
        return self._decode_predictions(
            predictions,
            lengths=lengths,
            ignore_ids=ignore_ids,
        )

    def decode_with_lengths(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor | None = None,
        ignore_ids: set[int] | None = None,
    ) -> list[list[int]]:
        """Length-aware greedy CTC decoding."""
        predictions = log_probs.argmax(dim=-1)
        return self._decode_predictions(
            predictions,
            lengths=lengths,
            ignore_ids=ignore_ids,
        )

    def _decode_predictions(
        self,
        predictions: torch.Tensor,
        lengths: torch.Tensor | None = None,
        ignore_ids: set[int] | None = None,
    ) -> list[list[int]]:
        decoded = []
        ignore_ids = {0} if ignore_ids is None else set(ignore_ids) | {0}

        for b in range(predictions.size(0)):
            seq = predictions[b]
            if lengths is not None:
                seq = seq[: lengths[b].item()]
            seq = seq.tolist()
            # Collapse repeats
            collapsed = []
            prev = -1
            for idx in seq:
                if idx != prev:
                    collapsed.append(idx)
                prev = idx
            collapsed = [idx for idx in collapsed if idx not in ignore_ids]
            decoded.append(collapsed)

        return decoded


class DualStreamCSLRModel(nn.Module):
    """Dual-stream CSLR model (Family B, §7.2).

    Two parallel streams process static pose (X) and motion (X_vel)
    features separately, then fuse before the CTC head.

    Args:
        pose_dim: Dimension of static pose features (104).
        motion_dim: Dimension of motion features (104).
        num_classes: Vocabulary size including CTC blank.
        conv_dim: Conv output channels per stream.
        conv_layers: Number of conv blocks per stream.
        conv_kernel_size: Conv kernel size.
        conv_dropout: Conv dropout.
        lstm_hidden_size: BiLSTM hidden size per direction.
        lstm_layers: Number of BiLSTM layers.
        lstm_dropout: BiLSTM dropout.
        fusion: Fusion strategy — 'concat' or 'gate'.
        multi_scale: Use multi-scale temporal conv.
        multi_scale_kernels: Kernel sizes for multi-scale.
    """

    def __init__(
        self,
        pose_dim: int = 104,
        motion_dim: int = 104,
        num_classes: int = 2000,
        conv_dim: int = 384,
        conv_layers: int = 3,
        conv_kernel_size: int = 5,
        conv_dropout: float = 0.1,
        lstm_hidden_size: int = 384,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        fusion: str = "concat",
        multi_scale: bool = True,
        multi_scale_kernels: list[int] | None = None,
    ):
        super().__init__()
        self.fusion = fusion

        ConvClass = MultiScaleTemporalConv if multi_scale else TemporalConvEncoder
        conv_kwargs = dict(
            conv_dim=conv_dim,
            dropout=conv_dropout,
            num_layers=conv_layers,
        )
        if multi_scale:
            conv_kwargs["kernel_sizes"] = multi_scale_kernels
        else:
            conv_kwargs["kernel_size"] = conv_kernel_size

        # Stream 1: Static pose
        self.pose_conv = ConvClass(input_dim=pose_dim, **conv_kwargs)
        self.pose_lstm = BiLSTMEncoder(
            input_dim=conv_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )

        # Stream 2: Motion
        self.motion_conv = ConvClass(input_dim=motion_dim, **conv_kwargs)
        self.motion_lstm = BiLSTMEncoder(
            input_dim=conv_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )

        encoder_out = 2 * lstm_hidden_size  # BiLSTM output per stream

        # Fusion layer
        if fusion == "concat":
            fused_dim = 2 * encoder_out
            self.fusion_proj = nn.Sequential(
                nn.Linear(fused_dim, encoder_out),
                nn.ReLU(inplace=True),
                nn.Dropout(lstm_dropout),
            )
            head_input_dim = encoder_out
        elif fusion == "gate":
            self.gate = nn.Sequential(
                nn.Linear(2 * encoder_out, encoder_out),
                nn.Sigmoid(),
            )
            head_input_dim = encoder_out
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        # CTC head
        self.ctc_head = CTCHead(
            input_dim=head_input_dim,
            num_classes=num_classes,
        )

    def forward(
        self,
        x_pose: torch.Tensor,
        x_motion: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_pose: (B, T, 104) static pose skeleton.
            x_motion: (B, T, 104) motion (velocity) features.
            lengths: (B,) original sequence lengths.

        Returns:
            (B, T, num_classes) log-probabilities for CTC.
        """
        # Stream 1: Pose
        p = self.pose_conv(x_pose)
        p = self.pose_lstm(p, lengths)

        # Stream 2: Motion
        m = self.motion_conv(x_motion)
        m = self.motion_lstm(m, lengths)

        # Fuse
        if self.fusion == "concat":
            fused = self.fusion_proj(torch.cat([p, m], dim=-1))
        elif self.fusion == "gate":
            gate_val = self.gate(torch.cat([p, m], dim=-1))
            fused = gate_val * p + (1 - gate_val) * m

        log_probs = self.ctc_head(fused)
        return log_probs

    def load_backbone(self, backbone_state_dict: dict, strict: bool = False):
        """Warm-start both streams from a single-stream ISLR backbone."""
        target_state = self.state_dict()
        remapped_state = {}
        for name, tensor in backbone_state_dict.items():
            if name.startswith("conv_encoder."):
                suffix = name[len("conv_encoder."):]
                pose_key = f"pose_conv.{suffix}"
                motion_key = f"motion_conv.{suffix}"
                if pose_key in target_state and target_state[pose_key].shape == tensor.shape:
                    remapped_state[pose_key] = tensor
                if motion_key in target_state and target_state[motion_key].shape == tensor.shape:
                    remapped_state[motion_key] = tensor.clone()
            elif name.startswith("seq_encoder."):
                suffix = name[len("seq_encoder."):]
                pose_key = f"pose_lstm.{suffix}"
                motion_key = f"motion_lstm.{suffix}"
                if pose_key in target_state and target_state[pose_key].shape == tensor.shape:
                    remapped_state[pose_key] = tensor
                if motion_key in target_state and target_state[motion_key].shape == tensor.shape:
                    remapped_state[motion_key] = tensor.clone()

        missing, unexpected = self.load_state_dict(remapped_state, strict=strict)
        if missing:
            print(f"Backbone loading — missing keys: {len(missing)}")
        if unexpected:
            print(f"Backbone loading — unexpected keys: {len(unexpected)}")
        return missing, unexpected

    def greedy_decode(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor | None = None,
        ignore_ids: set[int] | None = None,
    ) -> list[list[int]]:
        """Greedy CTC decoding."""
        predictions = log_probs.argmax(dim=-1)
        return self._decode_predictions(
            predictions,
            lengths=lengths,
            ignore_ids=ignore_ids,
        )

    def decode_with_lengths(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor | None = None,
        ignore_ids: set[int] | None = None,
    ) -> list[list[int]]:
        """Length-aware greedy CTC decoding."""
        predictions = log_probs.argmax(dim=-1)
        return self._decode_predictions(
            predictions,
            lengths=lengths,
            ignore_ids=ignore_ids,
        )

    def _decode_predictions(
        self,
        predictions: torch.Tensor,
        lengths: torch.Tensor | None = None,
        ignore_ids: set[int] | None = None,
    ) -> list[list[int]]:
        """Collapse repeats and remove blank/special tokens."""
        decoded = []
        ignore_ids = {0} if ignore_ids is None else set(ignore_ids) | {0}
        for b in range(predictions.size(0)):
            seq = predictions[b]
            if lengths is not None:
                seq = seq[: lengths[b].item()]
            seq = seq.tolist()
            collapsed = []
            prev = -1
            for idx in seq:
                if idx != prev:
                    collapsed.append(idx)
                prev = idx
            collapsed = [idx for idx in collapsed if idx not in ignore_ids]
            decoded.append(collapsed)
        return decoded
