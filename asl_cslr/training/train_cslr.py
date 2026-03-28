"""
CSLR training loop (Stage 2, §8.2).

Trains continuous sign language recognition with CTC loss,
including backbone initialization from ISLR and periodic WER evaluation.
"""

import logging
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from asl_cslr.data.dataset import CSLRDataset, cslr_collate_fn
from asl_cslr.data.augmentation import SkeletonAugmentor
from asl_cslr.data.vocab import GlossVocab
from asl_cslr.models.cslr_model import (
    CSLRModel,
    DualStreamCSLRModel,
    suppress_ctc_special_tokens,
)
from asl_cslr.utils.device import get_device, get_autocast_context
from asl_cslr.utils.model_config import (
    resolve_frame_feature_dim,
    resolve_motion_dim,
    resolve_single_stream_input_dim,
)
from .metrics import compute_cer, compute_wer
from .scheduler import build_scheduler

logger = logging.getLogger(__name__)
CHECKPOINT_VISIBILITY_TIMEOUT_SEC = 60.0
CHECKPOINT_VISIBILITY_POLL_SEC = 0.5


def _resolve_loader_workers(device: torch.device, requested_workers: int) -> int:
    """Choose a stable DataLoader worker count for the current device."""
    workers = max(0, int(requested_workers))
    if device.type == "mps" and workers > 0:
        logger.info(
            "Using num_workers=0 for CSLR DataLoaders on MPS for stability "
            "(requested %d)",
            workers,
        )
        return 0
    return workers


def _maybe_clear_mps_cache(
    device: torch.device,
    *,
    step: int | None = None,
    every_n_steps: int = 25,
):
    """Periodically clear cached MPS allocations during long training runs."""
    if device.type != "mps" or not hasattr(torch, "mps"):
        return
    if step is not None and step % every_n_steps != 0:
        return
    torch.mps.empty_cache()


def _wait_for_checkpoint_paths(
    paths: list[str | Path],
    *,
    timeout_sec: float = CHECKPOINT_VISIBILITY_TIMEOUT_SEC,
    poll_interval_sec: float = CHECKPOINT_VISIBILITY_POLL_SEC,
) -> None:
    """Wait for newly written checkpoints to become visible on disk."""
    normalized = [Path(path) for path in paths]
    deadline = time.monotonic() + max(timeout_sec, 0.0)
    missing = [str(path) for path in normalized if not path.exists()]
    while missing and time.monotonic() < deadline:
        time.sleep(max(poll_interval_sec, 0.01))
        missing = [str(path) for path in normalized if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Checkpoint files did not materialize before timeout: "
            f"{missing[:5]}"
        )


def _torch_load_with_retry(
    checkpoint_path: str | Path,
    *,
    map_location,
    weights_only: bool = False,
    timeout_sec: float = CHECKPOINT_VISIBILITY_TIMEOUT_SEC,
    poll_interval_sec: float = CHECKPOINT_VISIBILITY_POLL_SEC,
):
    """Load a checkpoint, retrying on transient visibility races."""
    normalized = Path(checkpoint_path)
    deadline = time.monotonic() + max(timeout_sec, 0.0)
    while True:
        try:
            return torch.load(
                normalized,
                map_location=map_location,
                weights_only=weights_only,
            )
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(max(poll_interval_sec, 0.01))


def _prune_epoch_checkpoints(save_dir: Path, keep_top_k: int):
    """Keep only the most recent epoch checkpoints; always preserve best.pt."""
    if keep_top_k <= 0:
        return
    checkpoints = sorted(
        save_dir.glob("epoch_*.pt"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    for ckpt_path in checkpoints[:-keep_top_k]:
        ckpt_path.unlink(missing_ok=True)


def _clear_stale_epoch_checkpoints(save_dir: Path):
    """Remove stale checkpoints from prior non-resumed runs."""
    for ckpt_path in save_dir.glob("epoch_*.pt"):
        ckpt_path.unlink(missing_ok=True)
    for checkpoint_name in ("best.pt", "last.pt"):
        (save_dir / checkpoint_name).unlink(missing_ok=True)


def _build_cslr_model(config: dict, num_classes: int, device: torch.device):
    input_dim = resolve_single_stream_input_dim(config["model"])
    common_kwargs = dict(
        num_classes=num_classes,
        conv_dim=config["model"]["conv_dim"],
        conv_layers=config["model"]["conv_layers"],
        conv_kernel_size=config["model"]["conv_kernel_size"],
        conv_dropout=config["model"]["conv_dropout"],
        lstm_hidden_size=config["model"]["lstm_hidden_size"],
        lstm_layers=config["model"]["lstm_layers"],
        lstm_dropout=config["model"]["lstm_dropout"],
        multi_scale=config["model"].get("multi_scale", False),
        multi_scale_kernels=config["model"].get("multi_scale_kernels"),
    )
    if config["model"].get("dual_stream", False):
        model = DualStreamCSLRModel(
            pose_dim=resolve_frame_feature_dim(config["model"]),
            motion_dim=resolve_motion_dim(config["model"]),
            fusion=config["model"].get("fusion", "concat"),
            **common_kwargs,
        )
    else:
        model = CSLRModel(
            input_dim=input_dim,
            encoder_type=config["model"].get("encoder_type", "bilstm"),
            transformer_hidden=config["model"].get("transformer_hidden", 256),
            transformer_layers=config["model"].get("transformer_layers", 4),
            transformer_heads=config["model"].get("transformer_heads", 4),
            **common_kwargs,
        )
    return model.to(device)


def _build_balanced_sampler(
    dataset: CSLRDataset,
    confusion_boost: dict[str, float] | None = None,
) -> WeightedRandomSampler:
    """Sample CSLR sequences inversely to their gloss coverage frequency.

    Args:
        dataset: The CSLR dataset.
        confusion_boost: Optional mapping of gloss -> extra weight multiplier
            for sequences containing glosses that are known to be weak/confused.
    """
    gloss_counts: Counter[str] = Counter()
    for entry in dataset.entries:
        gloss_counts.update(set(entry.get("glosses", [])))

    weights = []
    for entry in dataset.entries:
        glosses = set(entry.get("glosses", []))
        if not glosses:
            weights.append(1.0)
            continue
        weight = sum(1.0 / gloss_counts[gloss] for gloss in glosses) / len(glosses)
        # Boost sequences containing confused glosses
        if confusion_boost:
            boost = max(
                (confusion_boost.get(g, 1.0) for g in glosses),
                default=1.0,
            )
            weight *= boost
        weights.append(weight)

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def _initialize_ctc_head_biases(
    model: nn.Module,
    dataset: CSLRDataset,
    vocab: GlossVocab,
    *,
    blank_bias: float,
    smoothing: float,
    prior_scale: float,
) -> None:
    """Bias the CTC head away from early blank collapse.

    We initialize non-blank logits from gloss unigram priors and push the blank
    logit lower so the model starts by exploring real gloss emissions instead of
    decoding every sequence to empty.
    """
    ctc_head = getattr(model, "ctc_head", None)
    if ctc_head is None or not hasattr(ctc_head, "fc"):
        return

    gloss_indices = vocab.gloss_indices()
    if not gloss_indices:
        return

    gloss_counts: Counter[str] = Counter()
    for entry in dataset.entries:
        gloss_counts.update(entry.get("glosses", []))

    prior = torch.full((len(gloss_indices),), float(smoothing), dtype=torch.float32)
    for pos, token_idx in enumerate(gloss_indices):
        prior[pos] += float(gloss_counts.get(vocab.decode(token_idx), 0.0))

    gloss_bias = torch.log(prior)
    gloss_bias -= gloss_bias.mean()
    gloss_bias *= float(prior_scale)

    with torch.no_grad():
        ctc_head.fc.bias.zero_()
        ctc_head.fc.bias[vocab.blank_idx] = float(blank_bias)
        ctc_head.fc.bias[gloss_indices] = gloss_bias.to(
            ctc_head.fc.bias.device,
            dtype=ctc_head.fc.bias.dtype,
        )

    logger.info(
        "Initialized CTC head biases from train gloss priors (blank_bias=%.2f)",
        blank_bias,
    )


def _configure_ctc_blank_row(
    model: nn.Module,
    vocab: GlossVocab,
    *,
    blank_bias: float,
    zero_blank_weight: bool = False,
    special_bias: float | None = None,
) -> None:
    """Override the blank/special rows after any warm start or bias init."""
    ctc_head = getattr(model, "ctc_head", None)
    if ctc_head is None or not hasattr(ctc_head, "fc"):
        return

    with torch.no_grad():
        if zero_blank_weight:
            ctc_head.fc.weight[vocab.blank_idx].zero_()
        ctc_head.fc.bias[vocab.blank_idx] = float(blank_bias)

        if special_bias is not None:
            for idx in vocab.special_indices(include_blank=False):
                ctc_head.fc.bias[idx] = float(special_bias)
                if zero_blank_weight:
                    ctc_head.fc.weight[idx].zero_()


def _freeze_ctc_blank_gradients(model: nn.Module, vocab: GlossVocab) -> None:
    """Prevent the CTC blank row from taking over during the warmup phase."""
    ctc_head = getattr(model, "ctc_head", None)
    if ctc_head is None or not hasattr(ctc_head, "fc"):
        return

    if ctc_head.fc.weight.grad is not None:
        ctc_head.fc.weight.grad[vocab.blank_idx].zero_()
    if ctc_head.fc.bias.grad is not None:
        ctc_head.fc.bias.grad[vocab.blank_idx].zero_()


def _forward_cslr_model(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    use_amp: bool,
):
    """Run either single-stream or dual-stream CSLR forward pass."""
    input_lengths = batch["input_lengths"].to(device)
    if "features_pose" in batch:
        features_pose = batch["features_pose"].to(device)
        features_motion = batch["features_motion"].to(device)
        with get_autocast_context(device, use_amp):
            log_probs = model(features_pose, features_motion, input_lengths)
    else:
        features = batch["features"].to(device)
        with get_autocast_context(device, use_amp):
            log_probs = model(features, input_lengths)
    return log_probs, input_lengths


def _iter_conv_modules(model: nn.Module):
    """Yield conv modules that support freezing/unfreezing."""
    if hasattr(model, "conv_encoder"):
        yield model.conv_encoder
    if hasattr(model, "pose_conv"):
        yield model.pose_conv
    if hasattr(model, "motion_conv"):
        yield model.motion_conv


def _mask_ctc_special_log_probs(log_probs: torch.Tensor, vocab: GlossVocab) -> torch.Tensor:
    """Suppress non-blank special tokens before CTC loss or decoding."""
    return suppress_ctc_special_tokens(
        log_probs,
        vocab.special_indices(include_blank=False),
    )


def _compute_sample_confusion_weights(
    batch: dict,
    vocab: GlossVocab,
    confusion_boost: dict[str, float],
) -> torch.Tensor:
    """Compute per-sample loss weights based on confused glosses in each sequence."""
    label_lengths = batch["label_lengths"]
    labels = batch["labels"]
    batch_size = label_lengths.size(0)
    weights = torch.ones(batch_size, dtype=torch.float32)
    offset = 0
    for i in range(batch_size):
        ll = label_lengths[i].item()
        sample_ids = labels[offset: offset + ll].tolist()
        offset += ll
        max_boost = 1.0
        for token_id in sample_ids:
            gloss = vocab.decode(token_id)
            max_boost = max(max_boost, confusion_boost.get(gloss, 1.0))
        weights[i] = max_boost
    return weights


def train_cslr(config: dict, resume_path: str | None = None):
    """Run the full CSLR training pipeline.

    Args:
        config: Training configuration dictionary (from YAML).
    """
    # ---- Setup ----
    device = get_device()
    logger.info(f"Training CSLR on device: {device}")

    # Vocab
    vocab = GlossVocab.load(config["data"]["vocab_path"])
    num_classes = len(vocab)
    logger.info(f"Vocabulary size: {num_classes}")

    # Datasets
    expected_frame_feature_dim = resolve_frame_feature_dim(config["model"])
    required_schema_version = 2 if expected_frame_feature_dim == 208 else None

    aug_config = config.get("augmentation", {})
    augmentor = None
    if aug_config.get("enabled", False):
        augmentor = SkeletonAugmentor(
            spatial_jitter_std=aug_config.get("spatial_jitter_std", 0.008),
            scale_range=tuple(aug_config.get("scale_range", [0.95, 1.05])),
            rotation_range_deg=tuple(aug_config.get("rotation_range_deg", [0.0, 0.0])),
            pitch_range_deg=tuple(aug_config.get("pitch_range_deg", [0.0, 0.0])),
            yaw_range_deg=tuple(aug_config.get("yaw_range_deg", [0.0, 0.0])),
            translate_range=aug_config.get("translate_range", 0.03),
            temporal_crop_ratio=tuple(aug_config.get("temporal_crop_ratio", [0.9, 1.0])),
            temporal_drop_ratio=aug_config.get("temporal_drop_ratio", 0.0),
            flip_prob=aug_config.get("flip_prob", 0.0),
            allow_horizontal_flip=aug_config.get("allow_horizontal_flip", False),
            joint_dropout_prob=aug_config.get("joint_dropout_prob", 0.03),
            hand_dropout_prob=aug_config.get("hand_dropout_prob", 0.0),
            hand_dropout_ratio=tuple(aug_config.get("hand_dropout_ratio", [0.08, 0.25])),
            pose_dropout_prob=aug_config.get("pose_dropout_prob", 0.0),
            pose_dropout_ratio=tuple(aug_config.get("pose_dropout_ratio", [0.05, 0.18])),
            speed_perturb_range=tuple(aug_config.get("speed_perturb_range", [0.9, 1.1])),
            idle_hand_inject_prob=aug_config.get("idle_hand_inject_prob", 0.0),
            enabled=True,
        )
        logger.info("CSLR data augmentation enabled")

    train_dataset = CSLRDataset(
        manifest_path=config["data"]["train_manifest"],
        vocab=vocab,
        t_max=config["training"]["t_max"],
        use_motion=config["model"].get("use_motion", False),
        augmentor=augmentor,
        dual_stream=config["model"].get("dual_stream", False),
        frame_stride=config["data"].get("frame_stride", 1),
        expected_frame_feature_dim=expected_frame_feature_dim,
        required_schema_version=required_schema_version,
    )
    val_dataset = CSLRDataset(
        manifest_path=config["data"]["val_manifest"],
        vocab=vocab,
        t_max=config["training"]["t_max"],
        use_motion=config["model"].get("use_motion", False),
        dual_stream=config["model"].get("dual_stream", False),
        frame_stride=config["data"].get("frame_stride", 1),
        expected_frame_feature_dim=expected_frame_feature_dim,
        required_schema_version=required_schema_version,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("CSLR training split is empty after dataset preflight checks")
    if len(val_dataset) == 0:
        raise RuntimeError("CSLR validation split is empty after dataset preflight checks")

    sampler = None
    if config["training"].get("balanced_sampling", False):
        confusion_boost = config["training"].get("confusion_boost")
        sampler = _build_balanced_sampler(train_dataset, confusion_boost=confusion_boost)
        if confusion_boost:
            logger.info(
                "Using confusion-boosted balanced sampling: %s",
                confusion_boost,
            )
        else:
            logger.info("Using sequence-balanced sampling for CSLR training")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=cslr_collate_fn,
        num_workers=_resolve_loader_workers(
            device,
            config["data"].get("num_workers", 4),
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=cslr_collate_fn,
        num_workers=_resolve_loader_workers(
            device,
            config["data"].get("num_workers", 4),
        ),
    )

    # Model
    model = _build_cslr_model(config, num_classes, device)

    # Load pretrained weights: prefer CSLR checkpoint (full model), fall back to ISLR backbone
    pretrained_cslr_path = config["training"].get("pretrained_cslr")
    pretrained_path = config["training"].get("pretrained_backbone")
    loaded_from_cslr = False

    if pretrained_cslr_path and Path(pretrained_cslr_path).exists():
        logger.info(f"Loading full CSLR checkpoint from {pretrained_cslr_path}")
        ckpt = _torch_load_with_retry(
            pretrained_cslr_path,
            map_location=device,
            weights_only=False,
        )
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"],
            strict=False,
        )
        logger.info(
            "CSLR warm start loaded with %d missing and %d unexpected keys",
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.info("CSLR warm-start missing keys (sample): %s", missing[:4])
        loaded_from_cslr = True
    elif pretrained_path and Path(pretrained_path).exists():
        logger.info(f"Loading ISLR backbone from {pretrained_path}")
        ckpt = _torch_load_with_retry(
            pretrained_path,
            map_location=device,
            weights_only=False,
        )
        if hasattr(model, "load_backbone"):
            load_result = model.load_backbone(
                ckpt["model_state_dict"],
                strict=False,
            )
            if load_result is None:
                missing, unexpected = [], []
            else:
                missing, unexpected = load_result
            logger.info(
                "Backbone warm start loaded with %d missing and %d unexpected keys",
                len(missing),
                len(unexpected),
            )
            if missing:
                logger.info("Warm-start missing keys (sample): %s", missing[:4])
        else:
            missing, unexpected = model.load_state_dict(
                ckpt["model_state_dict"],
                strict=False,
            )
            logger.info(
                "Dual-stream warm start loaded with %d missing and %d unexpected keys",
                len(missing),
                len(unexpected),
            )
    else:
        logger.info("Training CSLR from scratch (no pretrained weights)")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # CTC bias init: skip when resuming or warm-starting from a CSLR checkpoint
    # (the CSLR checkpoint already has learned CTC head biases)
    blank_bias = config["training"].get("ctc_blank_bias_init")
    if (
        resume_path is None
        and not loaded_from_cslr
        and blank_bias is not None
    ):
        if not getattr(model, "_loaded_ctc_head_from_backbone", False):
            _initialize_ctc_head_biases(
                model,
                train_dataset,
                vocab,
                blank_bias=float(blank_bias),
                smoothing=float(config["training"].get("ctc_gloss_prior_smoothing", 1.0)),
                prior_scale=float(config["training"].get("ctc_gloss_prior_scale", 1.0)),
            )
        _configure_ctc_blank_row(
            model,
            vocab,
            blank_bias=float(blank_bias),
            zero_blank_weight=bool(
                config["training"].get("ctc_zero_blank_weight_on_init", False)
            ),
            special_bias=config["training"].get("ctc_special_bias_init"),
        )

    # CTC loss — use per-sample reduction when confusion weighting is active
    confusion_weights = config["training"].get("confusion_boost")
    use_per_sample_weighting = bool(confusion_weights)
    if use_per_sample_weighting:
        ctc_loss_fn = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True, reduction="none")
    else:
        ctc_loss_fn = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)
    # Validation always uses mean reduction
    ctc_loss_fn_val = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = build_scheduler(
        optimizer,
        scheduler_type=config["training"].get("scheduler", "cosine"),
        epochs=config["training"]["epochs"],
        warmup_epochs=config["training"].get("warmup_epochs", 5),
    )

    # Mixed precision (may exclude CTC loss computation)
    use_amp = config["training"].get("mixed_precision", True)
    amp_exclude_ctc = config["training"].get("amp_exclude_ctc", True)

    # Conv layer freezing
    freeze_conv_epochs = config["training"].get("freeze_conv_epochs", 0)

    # Logging
    save_dir = Path(config["checkpointing"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    if not resume_path:
        _clear_stale_epoch_checkpoints(save_dir)
    writer = SummaryWriter(config["logging"]["tensorboard_dir"])

    decode_every = config["evaluation"].get("decode_every_n_epochs", 5)

    # ---- Training loop ----
    best_val_wer = float("inf")
    best_val_loss = float("inf")
    start_epoch = 1
    epochs = config["training"]["epochs"]
    report_cer = "cer" in config["evaluation"].get("metrics", [])

    if resume_path:
        ckpt = _torch_load_with_retry(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_wer = ckpt.get("best_val_wer", ckpt.get("val_wer", float("inf")))
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(
            "Resumed CSLR checkpoint from %s at epoch %d",
            resume_path,
            start_epoch,
        )

    for epoch in range(start_epoch, epochs + 1):
        # Freeze/unfreeze conv layers
        if freeze_conv_epochs > 0:
            freeze = epoch <= freeze_conv_epochs
            for conv_module in _iter_conv_modules(model):
                first_block = None
                if hasattr(conv_module, "layers") and len(conv_module.layers) > 0:
                    first_block = conv_module.layers[0]
                elif hasattr(conv_module, "blocks") and len(conv_module.blocks) > 0:
                    first_block = conv_module.blocks[0]
                if first_block is not None:
                    for param in first_block.parameters():
                        param.requires_grad = not freeze

        model.train()
        total_loss = 0.0
        num_batches = 0
        t0 = time.time()
        freeze_blank_epochs = int(config["training"].get("freeze_blank_epochs", 0))

        for batch in train_loader:
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            optimizer.zero_grad()

            log_probs, input_lengths = _forward_cslr_model(
                model, batch, device, use_amp
            )
            log_probs = _mask_ctc_special_log_probs(log_probs, vocab)

            # CTC expects (T, B, C) format
            log_probs_ctc = log_probs.transpose(0, 1)

            # PyTorch MPS fallback for CTC loss often fails/is unsupported
            if device.type == "mps":
                log_probs_ctc_cpu = log_probs_ctc.cpu()
                labels_cpu = labels.cpu()
                input_lengths_cpu = input_lengths.cpu()
                label_lengths_cpu = label_lengths.cpu()
            else:
                log_probs_ctc_cpu = log_probs_ctc
                labels_cpu = labels
                input_lengths_cpu = input_lengths
                label_lengths_cpu = label_lengths

            if amp_exclude_ctc:
                loss = ctc_loss_fn(
                    log_probs_ctc_cpu.float(),
                    labels_cpu,
                    input_lengths_cpu,
                    label_lengths_cpu,
                ).to(device)
            else:
                with get_autocast_context(device, use_amp):
                    loss = ctc_loss_fn(
                        log_probs_ctc_cpu,
                        labels_cpu,
                        input_lengths_cpu,
                        label_lengths_cpu,
                    ).to(device)

            # Apply per-sample confusion weighting
            if use_per_sample_weighting and loss.dim() > 0:
                # Filter out NaN/Inf per-sample losses before weighting
                valid = ~(torch.isnan(loss) | torch.isinf(loss))
                if not valid.any():
                    logger.warning(f"All per-sample losses NaN/Inf at epoch {epoch}, skipping batch")
                    continue
                sample_weights = _compute_sample_confusion_weights(
                    batch, vocab, confusion_weights,
                )
                loss = (loss[valid] * sample_weights.to(device)[valid]).mean()

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss at epoch {epoch}, skipping batch")
                continue

            loss.backward()

            if epoch <= freeze_blank_epochs:
                _freeze_ctc_blank_gradients(model, vocab)

            grad_clip = config["training"].get("grad_clip_norm", 1.0)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            _maybe_clear_mps_cache(device, step=num_batches)

            total_loss += loss.item()
            num_batches += 1

            log_interval = config["logging"].get("log_every_n_steps", 50)
            if num_batches % log_interval == 0:
                cur_loss = total_loss / num_batches
                logger.info(f"Epoch {epoch} | Step {num_batches}/{len(train_loader)} | Loss: {cur_loss:.4f}")

        scheduler.step()
        _maybe_clear_mps_cache(device)

        train_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - t0

        # ---- Validation ----
        val_loss = _validate_cslr_loss(
            model, val_loader, ctc_loss_fn_val, vocab, device, use_amp, amp_exclude_ctc
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        log_msg = (
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Periodic WER evaluation. Always decode at least once so best.pt exists.
        should_decode = (
            epoch % decode_every == 0
            or (epoch == epochs and not (save_dir / "best.pt").exists())
        )
        if should_decode:
            val_wer, val_cer = _evaluate_cslr_wer(
                model, val_loader, vocab, device, use_amp
            )
            writer.add_scalar("WER/val", val_wer, epoch)
            if report_cer:
                writer.add_scalar("CER/val", val_cer, epoch)
            log_msg += f" | Val WER: {val_wer:.4f}"
            if report_cer:
                log_msg += f" | Val CER: {val_cer:.4f}"

            if val_wer < best_val_wer or (
                val_wer == best_val_wer and val_loss < best_val_loss
            ):
                best_val_wer = val_wer
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_wer": val_wer,
                    "val_cer": val_cer,
                    "best_val_wer": best_val_wer,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }, save_dir / "best.pt")
                _wait_for_checkpoint_paths([save_dir / "best.pt"])
                log_msg += " ★"

        logger.info(log_msg)

        # Periodic checkpointing
        if epoch % config["checkpointing"]["save_every_n_epochs"] == 0:
            epoch_path = save_dir / f"epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "best_val_wer": best_val_wer,
                "best_val_loss": best_val_loss,
                "config": config,
            }, epoch_path)
            _wait_for_checkpoint_paths([epoch_path])
            _prune_epoch_checkpoints(
                save_dir,
                config["checkpointing"].get("keep_top_k", 0),
            )

        last_path = save_dir / "last.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_wer": best_val_wer,
            "best_val_loss": best_val_loss,
            "config": config,
        }, last_path)
        _wait_for_checkpoint_paths([last_path])

    writer.close()
    logger.info(f"CSLR training complete. Best val WER: {best_val_wer:.4f}")


def _validate_cslr_loss(model, val_loader, ctc_loss_fn, vocab, device, use_amp, amp_exclude_ctc):
    """Compute validation CTC loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            log_probs, input_lengths = _forward_cslr_model(
                model, batch, device, use_amp
            )
            log_probs = _mask_ctc_special_log_probs(log_probs, vocab)

            log_probs_ctc = log_probs.transpose(0, 1)

            if device.type == "mps":
                log_probs_ctc_cpu = log_probs_ctc.cpu()
                labels_cpu = labels.cpu()
                input_lengths_cpu = input_lengths.cpu()
                label_lengths_cpu = label_lengths.cpu()
            else:
                log_probs_ctc_cpu = log_probs_ctc
                labels_cpu = labels
                input_lengths_cpu = input_lengths
                label_lengths_cpu = label_lengths

            if amp_exclude_ctc:
                loss = ctc_loss_fn(
                    log_probs_ctc_cpu.float(), labels_cpu, input_lengths_cpu, label_lengths_cpu
                ).to(device)
            else:
                loss = ctc_loss_fn(
                    log_probs_ctc_cpu, labels_cpu, input_lengths_cpu, label_lengths_cpu
                ).to(device)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                num_batches += 1
            _maybe_clear_mps_cache(device, step=num_batches)

    if num_batches == 0:
        raise RuntimeError("Validation produced no usable CSLR batches")

    _maybe_clear_mps_cache(device)
    return total_loss / num_batches


def _evaluate_cslr_wer(model, val_loader, vocab, device, use_amp):
    """Run greedy CTC decoding and compute WER on validation set."""
    model.eval()
    all_refs = []
    all_hyps = []
    ignore_ids = set(vocab.special_indices(include_blank=False))

    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"]
            label_lengths = batch["label_lengths"]

            log_probs, input_lengths = _forward_cslr_model(
                model, batch, device, use_amp
            )
            log_probs = _mask_ctc_special_log_probs(log_probs, vocab)

            # Greedy decode
            decoded = model.decode_with_lengths(
                log_probs,
                lengths=input_lengths,
                ignore_ids=ignore_ids,
            )

            # Reconstruct reference sequences from concatenated labels
            offset = 0
            for i in range(len(label_lengths)):
                ll = label_lengths[i].item()
                ref_ids = labels[offset: offset + ll].tolist()
                offset += ll
                all_refs.append(ref_ids)
                all_hyps.append(decoded[i])
            _maybe_clear_mps_cache(device, step=len(all_refs))

    if not all_refs:
        raise RuntimeError("Validation produced no CSLR reference sequences")

    # Compute WER
    total_wer = compute_wer(all_refs, all_hyps)
    total_cer = compute_cer(all_refs, all_hyps)
    _maybe_clear_mps_cache(device)
    return total_wer, total_cer
