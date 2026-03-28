#!/usr/bin/env python3
"""
CLI: Launch ISLR or CSLR training.

Usage:
    python scripts/train.py --config configs/islr_train.yaml
    python scripts/train.py --config configs/cslr_train.yaml
"""

import argparse
import logging

from asl_cslr.utils.logging import setup_logging
from asl_cslr.utils.io import load_yaml_config


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train ISLR or CSLR models."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to training config YAML file.",
    )
    parser.add_argument(
        "--resume",
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    setup_logging(level=getattr(logging, args.log_level))

    config = load_yaml_config(args.config)
    model_type = config["model"]["type"]

    logger.info(f"Starting {model_type.upper()} training with config: {args.config}")

    if model_type == "islr":
        from asl_cslr.training.train_islr import train_islr
        train_islr(config, resume_path=args.resume)
    elif model_type == "cslr":
        from asl_cslr.training.train_cslr import train_cslr
        train_cslr(config, resume_path=args.resume)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    main()
