"""Helpers for looking up local WLASL example clips for the web UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from asl_cslr.data.label_maps import clean_wlasl_gloss


@dataclass(frozen=True)
class SignExample:
    video_id: str
    gloss: str
    split: str
    signer_id: int | None
    video_path: str


class WlaslExampleIndex:
    """Load and query local WLASL example clips safely."""

    def __init__(
        self,
        mapping_path: str | Path | None = None,
        video_root: str | Path | None = None,
    ):
        repo_root = Path(__file__).resolve().parents[2]
        self._examples_by_gloss: dict[str, list[SignExample]] = {}
        self._examples_by_video_id: dict[str, SignExample] = {}
        self._sources = self._resolve_sources(repo_root, mapping_path, video_root)
        self._load()

    @staticmethod
    def _resolve_sources(
        repo_root: Path,
        mapping_path: str | Path | None,
        video_root: str | Path | None,
    ) -> list[tuple[Path, Path]]:
        if mapping_path is not None or video_root is not None:
            resolved_mapping = Path(
                mapping_path or repo_root / "data/raw/wlasl/wlasl_video_mapping.json"
            )
            resolved_video_root = Path(
                video_root or repo_root / "data/raw/wlasl/start_kit/raw_videos"
            )
            return [(resolved_mapping, resolved_video_root)]

        candidates = [
            (
                repo_root / "data/raw/kaggle/wlasl_processed/kaggle_expanded_mapping.json",
                repo_root / "data/raw/kaggle/wlasl_processed/videos",
            ),
            (
                repo_root / "data/raw/wlasl/wlasl_video_mapping.json",
                repo_root / "data/raw/wlasl/start_kit/raw_videos",
            ),
            (
                repo_root / "data/raw/kaggle/wlasl_processed/kaggle_shared_mapping.json",
                repo_root / "data/raw/kaggle/wlasl_processed/videos",
            ),
        ]
        return [(mapping, root) for mapping, root in candidates if mapping.exists()]

    @staticmethod
    def _split_rank(split: str) -> int:
        return {"train": 0, "val": 1, "test": 2}.get(split, 3)

    def _is_allowed_path(self, path: Path, allowed_root: Path) -> bool:
        try:
            return path.resolve().is_relative_to(allowed_root.resolve())
        except AttributeError:
            root = str(allowed_root.resolve())
            return str(path.resolve()).startswith(root)

    def _load(self) -> None:
        if not self._sources:
            raise FileNotFoundError("Missing WLASL example mappings")

        for mapping_path, video_root in self._sources:
            with open(mapping_path, "r", encoding="utf-8") as f:
                entries = json.load(f)

            for entry in entries:
                video_id = str(entry.get("video_id", "")).strip()
                gloss = clean_wlasl_gloss(entry.get("gloss", ""))
                split = str(entry.get("split", "train")).strip().lower() or "train"
                video_path = Path(entry.get("video_path", ""))
                signer_id = entry.get("signer_id")

                if not video_id or not gloss or not video_path.exists():
                    continue
                if video_id in self._examples_by_video_id:
                    continue
                if not self._is_allowed_path(video_path, video_root):
                    continue

                example = SignExample(
                    video_id=video_id,
                    gloss=gloss,
                    split=split,
                    signer_id=int(signer_id) if signer_id is not None else None,
                    video_path=str(video_path.resolve()),
                )
                self._examples_by_gloss.setdefault(gloss, []).append(example)
                self._examples_by_video_id[video_id] = example

        for gloss, examples in self._examples_by_gloss.items():
            examples.sort(
                key=lambda ex: (
                    self._split_rank(ex.split),
                    ex.signer_id if ex.signer_id is not None else 10**9,
                    ex.video_id,
                )
            )

    def list_examples(self, gloss: str, limit: int = 3) -> list[SignExample]:
        canonical = clean_wlasl_gloss(gloss)
        return list(self._examples_by_gloss.get(canonical, []))[:limit]

    def resolve_video_path(self, video_id: str) -> Path | None:
        example = self._examples_by_video_id.get(str(video_id))
        if example is None:
            return None
        return Path(example.video_path)

    def supported_glosses(self) -> list[str]:
        return sorted(self._examples_by_gloss.keys())
