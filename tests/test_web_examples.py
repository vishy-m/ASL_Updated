import json

import pytest

from asl_cslr.online.sign_examples import WlaslExampleIndex
from asl_cslr.online.web_server import ASLWebServer


def _make_minimal_online_config():
    return {
        "default_mode": "islr",
        "camera": {
            "device_id": 0,
            "capture_fps": 30,
            "downsample_factor": 2,
        },
        "islr": {
            "enabled": True,
            "effective_fps": 15,
            "window_duration_sec": 2.0,
            "hop_duration_sec": 0.5,
            "stability_windows": 2,
            "confidence_threshold": 0.6,
        },
        "cslr": {
            "enabled": False,
            "decode_interval_sec": 0.5,
            "buffer_duration_sec": 3.0,
        },
    }


def test_wlasl_example_index_lists_local_clip(tmp_path):
    video_root = tmp_path / "raw_videos"
    video_root.mkdir()
    video_path = video_root / "12345.mp4"
    video_path.write_bytes(b"fake-mp4")

    mapping_path = tmp_path / "wlasl_video_mapping.json"
    mapping = [
        {
            "video_id": "12345",
            "video_path": str(video_path),
            "gloss": "book",
            "split": "train",
            "signer_id": 7,
        }
    ]
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    index = WlaslExampleIndex(mapping_path=mapping_path, video_root=video_root)
    examples = index.list_examples("BOOK")

    assert len(examples) == 1
    assert examples[0].video_id == "12345"
    assert index.resolve_video_path("12345") == video_path.resolve()


def test_web_server_exposes_example_api_and_media_route(tmp_path):
    video_root = tmp_path / "raw_videos"
    video_root.mkdir()
    video_path = video_root / "54321.mp4"
    video_path.write_bytes(b"fake-mp4")

    mapping_path = tmp_path / "wlasl_video_mapping.json"
    mapping = [
        {
            "video_id": "54321",
            "video_path": str(video_path),
            "gloss": "like",
            "split": "val",
            "signer_id": 9,
        }
    ]
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    example_index = WlaslExampleIndex(mapping_path=mapping_path, video_root=video_root)
    server = ASLWebServer(
        _make_minimal_online_config(),
        mode="islr",
        port=0,
        load_models=False,
        example_index=example_index,
    )
    client = server.app.test_client()

    api_response = client.get("/api/sign-examples/LIKE")
    assert api_response.status_code == 200
    payload = api_response.get_json()
    assert payload["canonical_gloss"] == "LIKE"
    assert payload["count"] == 1
    assert payload["examples"][0]["video_id"] == "54321"
    assert payload["examples"][0]["video_url"] == "/media/wlasl/54321"

    media_response = client.get("/media/wlasl/54321")
    assert media_response.status_code == 200
    assert media_response.data == b"fake-mp4"
    assert media_response.mimetype == "video/mp4"

    assert client.get("/media/wlasl/does-not-exist").status_code == 404


def test_wlasl_example_index_merges_multiple_sources_without_duplicate_video_ids(
    tmp_path, monkeypatch
):
    root_a = tmp_path / "source_a"
    root_b = tmp_path / "source_b"
    root_a.mkdir()
    root_b.mkdir()

    video_a = root_a / "12345.mp4"
    video_b_dup = root_b / "12345.mp4"
    video_b_new = root_b / "99999.mp4"
    video_a.write_bytes(b"a")
    video_b_dup.write_bytes(b"b")
    video_b_new.write_bytes(b"c")

    mapping_a = tmp_path / "mapping_a.json"
    mapping_b = tmp_path / "mapping_b.json"
    mapping_a.write_text(
        json.dumps(
            [
                {
                    "video_id": "12345",
                    "video_path": str(video_a),
                    "gloss": "dog",
                    "split": "train",
                    "signer_id": 1,
                }
            ]
        ),
        encoding="utf-8",
    )
    mapping_b.write_text(
        json.dumps(
            [
                {
                    "video_id": "12345",
                    "video_path": str(video_b_dup),
                    "gloss": "dog",
                    "split": "val",
                    "signer_id": 9,
                },
                {
                    "video_id": "99999",
                    "video_path": str(video_b_new),
                    "gloss": "apple",
                    "split": "test",
                    "signer_id": 4,
                },
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        WlaslExampleIndex,
        "_resolve_sources",
        staticmethod(lambda *_args, **_kwargs: [(mapping_a, root_a), (mapping_b, root_b)]),
    )

    index = WlaslExampleIndex()

    dog_examples = index.list_examples("DOG")
    apple_examples = index.list_examples("APPLE")

    assert len(dog_examples) == 1
    assert dog_examples[0].video_path == str(video_a.resolve())
    assert len(apple_examples) == 1
    assert apple_examples[0].video_id == "99999"
