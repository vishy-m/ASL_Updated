#!/usr/bin/env python3
"""Download and map selected WLASL glosses for focused CSLR experiments.

This script recovers additional raw WLASL clips for a chosen vocabulary,
skipping legacy SWF-only sources, and writes a deterministic mapping JSON that
can be fed directly into `scripts/preprocess_wlasl_batch.py`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from asl_cslr.data.label_maps import clean_wlasl_gloss

logger = logging.getLogger(__name__)

YOUTUBE_DOMAINS = ("youtube.com", "youtu.be")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".avi")


def _resolve_path(path_like: str | Path, base_dir: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (base_dir / path)


def _load_glosses(args: argparse.Namespace) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    glosses = list(args.glosses or [])
    if args.gloss_file:
        for line in Path(args.gloss_file).read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            glosses.append(text)

    for gloss in glosses:
        canonical = clean_wlasl_gloss(gloss)
        if not canonical or canonical in seen:
            continue
        normalized.append(canonical)
        seen.add(canonical)
    return normalized


def _normalize_metadata_gloss(gloss: str) -> str:
    return clean_wlasl_gloss(gloss)


def _iter_selected_instances(metadata: list[dict[str, Any]], glosses: set[str]):
    for item in metadata:
        gloss = _normalize_metadata_gloss(str(item.get("gloss", "")))
        if gloss not in glosses:
            continue
        for inst in item.get("instances", []):
            yield gloss, inst


def _find_existing_video(raw_videos_dir: Path, video_id: str) -> Path | None:
    for ext in VIDEO_EXTENSIONS:
        candidate = raw_videos_dir / f"{video_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _is_youtube(url: str) -> bool:
    lower = url.lower()
    return any(domain in lower for domain in YOUTUBE_DOMAINS)


def _is_downloadable(url: str) -> bool:
    return bool(url) and not url.lower().endswith(".swf")


def _download_direct(url: str, target_path: Path, timeout_sec: int) -> None:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "-k",
        "-A",
        "Mozilla/5.0",
        "--max-time",
        str(timeout_sec),
        "-o",
        str(target_path),
    ]
    if "aslbricks.org" in domain:
        cmd.extend(["-e", "http://aslbricks.org/"])
    elif "aslsignbank.haskins.yale.edu" in domain:
        cmd.extend(["-e", "https://aslsignbank.haskins.yale.edu/"])
    elif "spreadthesign.com" in domain:
        cmd.extend(["-e", "https://www.spreadthesign.com/"])
    elif "startasl.com" in domain:
        cmd.extend(["-e", "https://www.startasl.com/"])
    cmd.append(url)
    subprocess.run(cmd, check=True)


def _download_youtube(url: str, video_id: str, raw_videos_dir: Path) -> None:
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "--recode-video",
        "mp4",
        "-o",
        str(raw_videos_dir / f"{video_id}.%(ext)s"),
        url,
    ]
    subprocess.run(cmd, check=True)


def _download_instance(task: dict[str, Any]) -> tuple[str, str, str]:
    """Return (status, video_id, message)."""
    video_id = str(task["video_id"])
    url = str(task["url"])
    raw_videos_dir = Path(task["raw_videos_dir"])
    timeout_sec = int(task["timeout_sec"])

    existing = _find_existing_video(raw_videos_dir, video_id)
    if existing is not None:
        return "cached", video_id, str(existing)

    try:
        if _is_youtube(url):
            _download_youtube(url, video_id, raw_videos_dir)
        else:
            target_path = raw_videos_dir / f"{video_id}.mp4"
            _download_direct(url, target_path, timeout_sec)
        existing = _find_existing_video(raw_videos_dir, video_id)
        if existing is None:
            return "failed", video_id, "download finished but no local file was found"
        return "downloaded", video_id, str(existing)
    except subprocess.CalledProcessError as exc:
        return "failed", video_id, f"{exc}"


def _build_mapping(
    metadata: list[dict[str, Any]],
    glosses: set[str],
    raw_videos_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mapping: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for gloss, inst in _iter_selected_instances(metadata, glosses):
        video_id = str(inst.get("video_id", ""))
        video_path = _find_existing_video(raw_videos_dir, video_id)
        entry = {
            "video_id": video_id,
            "video_path": str(video_path) if video_path is not None else "",
            "gloss": gloss,
            "split": inst.get("split", "train"),
            "fps": inst.get("fps"),
            "frame_start": int(inst.get("frame_start", 0) or 0),
            "frame_end": int(inst.get("frame_end", -1) or -1),
            "signer_id": inst.get("signer_id"),
            "source": inst.get("source"),
            "url": inst.get("url"),
        }
        if video_path is None:
            missing.append(entry)
        else:
            mapping.append(entry)

    mapping.sort(
        key=lambda item: (
            item.get("split", ""),
            item.get("gloss", ""),
            item.get("video_id", ""),
        )
    )
    missing.sort(
        key=lambda item: (
            item.get("gloss", ""),
            item.get("video_id", ""),
        )
    )
    return mapping, missing


def main():
    parser = argparse.ArgumentParser(
        description="Download additional WLASL videos for selected glosses.",
    )
    parser.add_argument(
        "--metadata",
        default="data/raw/wlasl/start_kit/WLASL_v0.3.json",
        help="Path to the full WLASL metadata JSON.",
    )
    parser.add_argument(
        "--raw-videos-dir",
        default="data/raw/wlasl/start_kit/raw_videos",
        help="Directory containing downloaded raw WLASL clips.",
    )
    parser.add_argument(
        "--mapping-out",
        default="data/raw/wlasl/live_wide_mapping.json",
        help="Where to write the deterministic selected-gloss mapping JSON.",
    )
    parser.add_argument(
        "--summary-out",
        default="data/raw/wlasl/live_wide_download_summary.json",
        help="Where to write the download summary JSON.",
    )
    parser.add_argument(
        "--glosses",
        nargs="*",
        default=None,
        help="Glosses to download.",
    )
    parser.add_argument(
        "--gloss-file",
        default="configs/live_wide_glosses.txt",
        help="Optional newline-delimited gloss list.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent download workers.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="Per-file timeout for direct HTTP downloads.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only rebuild mapping/summary from already downloaded files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[1]
    metadata_path = _resolve_path(args.metadata, repo_root)
    raw_videos_dir = _resolve_path(args.raw_videos_dir, repo_root)
    mapping_out = _resolve_path(args.mapping_out, repo_root)
    summary_out = _resolve_path(args.summary_out, repo_root)
    raw_videos_dir.mkdir(parents=True, exist_ok=True)

    glosses = _load_glosses(args)
    if not glosses:
        raise ValueError("No glosses provided")
    selected = set(glosses)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    tasks: list[dict[str, Any]] = []
    by_gloss_missing: Counter[str] = Counter()
    skipped_nondownloadable: Counter[str] = Counter()

    for gloss, inst in _iter_selected_instances(metadata, selected):
        video_id = str(inst.get("video_id", ""))
        if _find_existing_video(raw_videos_dir, video_id) is not None:
            continue
        url = str(inst.get("url", ""))
        if not _is_downloadable(url):
            skipped_nondownloadable[gloss] += 1
            continue
        tasks.append(
            {
                "gloss": gloss,
                "video_id": video_id,
                "url": url,
                "raw_videos_dir": str(raw_videos_dir),
                "timeout_sec": args.timeout_sec,
            }
        )
        by_gloss_missing[gloss] += 1

    logger.info(
        "Selected %d glosses with %d downloadable missing clips",
        len(glosses),
        len(tasks),
    )
    for gloss in glosses:
        logger.info(
            "%s: missing downloadable=%d skipped_swf=%d",
            gloss,
            by_gloss_missing.get(gloss, 0),
            skipped_nondownloadable.get(gloss, 0),
        )

    results: list[tuple[str, str, str]] = []
    if not args.skip_download and tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(_download_instance, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                status, video_id, message = future.result()
                results.append((status, video_id, message))
                if status == "failed":
                    logger.warning("Failed %s: %s", video_id, message)
    else:
        logger.info("Skipping downloads; rebuilding mapping from local files only")

    mapping, missing = _build_mapping(metadata, selected, raw_videos_dir)
    mapping_out.parent.mkdir(parents=True, exist_ok=True)
    mapping_out.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")

    status_counts = Counter(status for status, _, _ in results)
    mapped_counts = Counter(entry["gloss"] for entry in mapping)
    missing_counts = Counter(entry["gloss"] for entry in missing)
    summary = {
        "glosses": glosses,
        "download_status_counts": dict(sorted(status_counts.items())),
        "mapped_clip_counts": dict(sorted(mapped_counts.items())),
        "missing_clip_counts": dict(sorted(missing_counts.items())),
        "skipped_nondownloadable_counts": dict(sorted(skipped_nondownloadable.items())),
        "failed_downloads": [
            {"video_id": video_id, "message": message}
            for status, video_id, message in results
            if status == "failed"
        ],
        "mapping_path": str(mapping_out),
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    logger.info("Wrote mapping: %s (%d entries)", mapping_out, len(mapping))
    logger.info("Wrote summary: %s", summary_out)
    if missing:
        logger.warning("Still missing %d selected clips after download", len(missing))


if __name__ == "__main__":
    main()
