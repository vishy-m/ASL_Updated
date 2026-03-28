"""
ISLR training loop (Stage 1, §8.1).

Trains isolated sign recognition with cross-entropy loss,
mixed precision on MPS, and TensorBoard logging.
"""

import logging
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from asl_cslr.data.dataset import ISLRDataset, islr_collate_fn
from asl_cslr.data.vocab import GlossVocab
from asl_cslr.data.augmentation import SkeletonAugmentor
from asl_cslr.models.islr_model import ISLRModel
from asl_cslr.utils.device import get_device, get_autocast_context
from asl_cslr.utils.model_config import (
    resolve_frame_feature_dim,
    resolve_single_stream_input_dim,
)
from .metrics import compute_accuracy, macro_averaged_accuracy
from .scheduler import build_scheduler

logger = logging.getLogger(__name__)


def _resolve_loader_workers(device: torch.device, requested_workers: int) -> int:
    """Choose a stable DataLoader worker count for the current device."""
    workers = max(0, int(requested_workers))
    if device.type == "mps" and workers > 0:
        logger.info(
            "Using num_workers=0 for ISLR DataLoaders on MPS for stability "
            "(requested %d)",
            workers,
        )
        return 0
    return workers


def _mask_special_logits(logits: torch.Tensor, vocab: GlossVocab) -> torch.Tensor:
    """Exclude reserved tokens from isolated-sign classification."""
    masked = logits.clone()
    special_indices = vocab.special_indices(include_blank=True)
    if special_indices:
        masked[:, special_indices] = torch.finfo(masked.dtype).min
    return masked


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


def _build_balanced_sampler(dataset: ISLRDataset, vocab: GlossVocab) -> WeightedRandomSampler:
    """Sample training clips inversely proportional to class frequency."""
    label_ids = []
    for entry in dataset.entries:
        glosses = entry.get("glosses", [])
        gloss = glosses[0] if glosses else "<unk>"
        label_ids.append(vocab.encode(gloss))

    counts = Counter(label_ids)
    weights = torch.as_tensor(
        [1.0 / counts[label_id] for label_id in label_ids],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(label_ids),
        replacement=True,
    )


def train_islr(config: dict, resume_path: str | None = None):
    """Run the full ISLR training pipeline.

    Args:
        config: Training configuration dictionary (from YAML).
    """
    # ---- Setup ----
    device = get_device()
    logger.info(f"Training on device: {device}")

    # Vocab
    vocab = GlossVocab.load(config["data"]["vocab_path"])
    num_classes = len(vocab)
    logger.info(f"Vocabulary size: {num_classes}")

    # Datasets
    expected_frame_feature_dim = resolve_frame_feature_dim(config["model"])
    required_schema_version = 2 if expected_frame_feature_dim == 208 else None

    # Build augmentor for training (disabled for validation)
    aug_config = config.get("augmentation", {})
    augmentor = None
    if aug_config.get("enabled", False):
        augmentor = SkeletonAugmentor(
            spatial_jitter_std=aug_config.get("spatial_jitter_std", 0.01),
            scale_range=tuple(aug_config.get("scale_range", [0.9, 1.1])),
            rotation_range_deg=tuple(aug_config.get("rotation_range_deg", [0.0, 0.0])),
            pitch_range_deg=tuple(aug_config.get("pitch_range_deg", [0.0, 0.0])),
            yaw_range_deg=tuple(aug_config.get("yaw_range_deg", [0.0, 0.0])),
            translate_range=aug_config.get("translate_range", 0.05),
            temporal_crop_ratio=tuple(aug_config.get("temporal_crop_ratio", [0.8, 1.0])),
            temporal_drop_ratio=aug_config.get("temporal_drop_ratio", 0.0),
            flip_prob=aug_config.get("flip_prob", 0.0),
            allow_horizontal_flip=aug_config.get("allow_horizontal_flip", False),
            joint_dropout_prob=aug_config.get("joint_dropout_prob", 0.05),
            hand_dropout_prob=aug_config.get("hand_dropout_prob", 0.0),
            hand_dropout_ratio=tuple(aug_config.get("hand_dropout_ratio", [0.08, 0.25])),
            pose_dropout_prob=aug_config.get("pose_dropout_prob", 0.0),
            pose_dropout_ratio=tuple(aug_config.get("pose_dropout_ratio", [0.05, 0.18])),
            speed_perturb_range=tuple(aug_config.get("speed_perturb_range", [0.8, 1.2])),
            idle_hand_inject_prob=aug_config.get("idle_hand_inject_prob", 0.0),
            enabled=True,
        )
        logger.info("Data augmentation enabled")

    train_dataset = ISLRDataset(
        manifest_path=config["data"]["train_manifest"],
        vocab=vocab,
        use_motion=config["model"].get("use_motion", False),
        augmentor=augmentor,
        expected_frame_feature_dim=expected_frame_feature_dim,
        required_schema_version=required_schema_version,
    )
    val_dataset = ISLRDataset(
        manifest_path=config["data"]["val_manifest"],
        vocab=vocab,
        use_motion=config["model"].get("use_motion", False),
        # No augmentor for validation
        expected_frame_feature_dim=expected_frame_feature_dim,
        required_schema_version=required_schema_version,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("ISLR training split is empty")
    if len(val_dataset) == 0:
        raise RuntimeError("ISLR validation split is empty")

    sampler = None
    if config["training"].get("balanced_sampling", False):
        sampler = _build_balanced_sampler(train_dataset, vocab)
        logger.info("Using class-balanced sampling for ISLR training")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=islr_collate_fn,
        num_workers=_resolve_loader_workers(
            device,
            config["data"].get("num_workers", 4),
        ),
        pin_memory=config["data"].get("pin_memory", False),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=islr_collate_fn,
        num_workers=_resolve_loader_workers(
            device,
            config["data"].get("num_workers", 4),
        ),
    )

    # Model
    input_dim = resolve_single_stream_input_dim(config["model"])
    model = ISLRModel(
        input_dim=input_dim,
        num_classes=num_classes,
        conv_dim=config["model"]["conv_dim"],
        conv_layers=config["model"]["conv_layers"],
        conv_kernel_size=config["model"]["conv_kernel_size"],
        conv_dropout=config["model"]["conv_dropout"],
        encoder_type=config["model"].get("encoder_type", "bilstm"),
        lstm_hidden_size=config["model"]["lstm_hidden_size"],
        lstm_layers=config["model"]["lstm_layers"],
        lstm_dropout=config["model"]["lstm_dropout"],
        fc_dropout=config["model"]["fc_dropout"],
        pool=config["model"].get("pool", "last"),
        multi_scale=config["model"].get("multi_scale", False),
        multi_scale_kernels=config["model"].get("multi_scale_kernels"),
    ).to(device)

    pretrained_path = config["training"].get("pretrained_backbone")
    if pretrained_path and Path(pretrained_path).exists():
        logger.info("Loading ISLR backbone from %s", pretrained_path)
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        backbone_state = {
            name: tensor
            for name, tensor in ckpt["model_state_dict"].items()
            if name.startswith("conv_encoder.") or name.startswith("seq_encoder.")
        }
        missing, unexpected = model.load_state_dict(backbone_state, strict=False)
        logger.info(
            "Warm start loaded with %d missing and %d unexpected keys",
            len(missing),
            len(unexpected),
        )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
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

    # Mixed precision
    use_amp = config["training"].get("mixed_precision", True)

    # Logging
    save_dir = Path(config["checkpointing"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    if not resume_path:
        _clear_stale_epoch_checkpoints(save_dir)
    writer = SummaryWriter(config["logging"]["tensorboard_dir"])

    # ---- Training loop ----
    best_val_acc = 0.0
    best_val_loss = float("inf")
    start_epoch = 1
    epochs = config["training"]["epochs"]
    selection_metric = config["checkpointing"].get("selection_metric", "top1_accuracy")

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_acc = ckpt.get("best_val_acc", ckpt.get("val_acc", 0.0))
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(
            "Resumed ISLR checkpoint from %s at epoch %d",
            resume_path,
            start_epoch,
        )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        t0 = time.time()

        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer.zero_grad()

            with get_autocast_context(device, use_amp):
                logits = model(features, lengths)
                masked_logits = _mask_special_logits(logits, vocab)
                loss = criterion(masked_logits, labels)

            loss.backward()

            # Gradient clipping
            grad_clip = config["training"].get("grad_clip_norm", 1.0)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (masked_logits.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

        scheduler.step()

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        elapsed = time.time() - t0

        # ---- Validation ----
        val_loss, val_acc, val_top5, val_macro = _validate_islr(
            model, val_loader, criterion, device, use_amp, vocab
        )

        # Log
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val_top1", val_acc, epoch)
        writer.add_scalar("Accuracy/val_top5", val_top5, epoch)
        writer.add_scalar("Accuracy/val_macro", val_macro, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Top1: {val_acc:.4f} "
            f"Top5: {val_top5:.4f} Macro: {val_macro:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Checkpointing
        if epoch % config["checkpointing"]["save_every_n_epochs"] == 0:
            ckpt_path = save_dir / f"epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro": val_macro,
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "config": config,
            }, ckpt_path)
            _prune_epoch_checkpoints(
                save_dir,
                config["checkpointing"].get("keep_top_k", 0),
            )

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro": val_macro,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "config": config,
        }, save_dir / "last.pt")

        selection_value = val_macro if selection_metric == "macro_accuracy" else val_acc
        best_so_far = best_val_acc
        if selection_value > best_so_far or (
            selection_value == best_so_far and val_loss < best_val_loss
        ):
            best_val_acc = selection_value
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro": val_macro,
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "config": config,
            }, save_dir / "best.pt")
            logger.info(
                "  ★ New best %s: %.4f",
                selection_metric,
                best_val_acc,
            )

    writer.close()
    logger.info(f"Training complete. Best {selection_metric}: {best_val_acc:.4f}")


def _validate_islr(model, val_loader, criterion, device, use_amp, vocab):
    """Run validation and return loss, top-1, and top-5 accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct_1 = 0
    total_correct_5 = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            with get_autocast_context(device, use_amp):
                logits = model(features, lengths)
                masked_logits = _mask_special_logits(logits, vocab)
                loss = criterion(masked_logits, labels)

            total_loss += loss.item() * labels.size(0)
            acc1, acc5 = compute_accuracy(masked_logits, labels, topk=(1, 5))
            total_correct_1 += acc1 * labels.size(0)
            total_correct_5 += acc5 * labels.size(0)
            total_samples += labels.size(0)
            all_preds.extend(masked_logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    if total_samples == 0:
        raise RuntimeError("ISLR validation produced zero samples")
    macro = macro_averaged_accuracy(all_preds, all_labels, num_classes=len(vocab))

    return (
        total_loss / total_samples,
        total_correct_1 / total_samples,
        total_correct_5 / total_samples,
        macro,
    )
