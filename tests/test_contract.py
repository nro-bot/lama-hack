"""Pin the client's constants to the checkpoint's declared contract.

These constants are invisible coupling: the embodiment resizes frames and slices
chunks client-side, and nothing at runtime complains if those numbers disagree
with what the policy was trained on. The model just quietly receives the wrong
thing and produces subtly wrong actions — which, at the servos, is a real arm
moving to a wrong pose.

We already got this wrong once: the vendored embodiment carried the newt hosted
base model's 378px and a chunk_size guess of 10, while play_xylophone_100
declares 224px and 30. This file exists so the next mismatch fails at pytest.

Values below are read from the checkpoint's config.json, fetched from the Hub
and cached. Skipped when offline or when huggingface_hub isn't installed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

import embodiment

hf_hub = pytest.importorskip("huggingface_hub")

CHECKPOINT = "ArjunPrasaath/play_xylophone_100"


@pytest.fixture(scope="module")
def config() -> dict:
    try:
        path = hf_hub.hf_hub_download(CHECKPOINT, "config.json")
    except Exception as exc:
        pytest.skip(f"cannot reach {CHECKPOINT} ({type(exc).__name__})")
    return json.load(open(path))


def test_image_size_matches_the_declared_input_features(config: dict) -> None:
    """The bug this file was written for."""
    for cam in ("cam0", "cam1"):
        shape = config["input_features"][f"observation.images.{cam}"]["shape"]
        _, height, width = shape
        assert height == width, f"{cam} is not square: {shape}"
        assert height == embodiment._IMAGE_SIZE, (
            f"{cam} expects {height}px but embodiment._IMAGE_SIZE is "
            f"{embodiment._IMAGE_SIZE}. The rig would send the wrong size."
        )


def test_chunk_truncation_is_within_the_trained_horizon(config: dict) -> None:
    """Slicing more actions than exist is silently a no-op; flag the confusion."""
    chunk_size = config["chunk_size"]
    assert embodiment.MAX_ACTIONS_PER_CHUNK <= chunk_size, (
        f"MAX_ACTIONS_PER_CHUNK={embodiment.MAX_ACTIONS_PER_CHUNK} exceeds the "
        f"trained chunk_size={chunk_size}; the extra is meaningless"
    )
    assert embodiment.MAX_ACTIONS_PER_CHUNK > 0


def test_state_and_action_are_six_dof(config: dict) -> None:
    assert config["input_features"]["observation.state"]["shape"] == [
        len(embodiment._JOINT_ORDER)
    ]
    assert config["output_features"]["action"]["shape"] == [
        len(embodiment._JOINT_ORDER)
    ]


def test_camera_rename_targets_the_declared_feature_names(config: dict) -> None:
    """top/side -> cam0/cam1 must match what the policy actually declares.

    Get this backwards and the model sees the side view where it expects the
    top one — which reads as a badly trained policy rather than a wiring bug.
    """
    from modal_ws import _CAMERA_RENAME

    assert list(_CAMERA_RENAME) == embodiment._CAMERA_KEYS
    for renamed in _CAMERA_RENAME.values():
        assert f"observation.images.{renamed}" in config["input_features"]


def test_normalization_requires_the_saved_processors(config: dict) -> None:
    """QUANTILES stats live in the processor files, so they cannot be optional.

    If this ever becomes IDENTITY/MEAN_STD the mandatory-preprocessor stance in
    modal_ws._load_policy could be relaxed — until then, don't.
    """
    mapping = config["normalization_mapping"]
    assert mapping["STATE"] == "QUANTILES"
    assert mapping["ACTION"] == "QUANTILES"
