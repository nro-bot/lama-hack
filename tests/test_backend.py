"""Backend selection tests — no hardware, no network.

_resolve_backend guards the environment hazards that make the two inference
backends interfere with each other. The dangerous one: NT_INFERENCE_URL
overrides registry discovery for ALL models in the newt SDK, so a leftover
export from a Modal session would make a "hosted newt" run silently talk to
Modal — same protocol, wrong model, no error anywhere.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run as run_mod
from embodiment import _IMAGE_SIZE, _NEWT_IMAGE_SIZE


def _args(backend: str = "auto", model: str = "so101") -> argparse.Namespace:
    return argparse.Namespace(backend=backend, model=model)


URL = "wss://example--xylophone.modal.run"


def test_auto_picks_modal_when_url_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NT_INFERENCE_URL", URL)
    monkeypatch.delenv("NT_API_KEY", raising=False)

    image_size, model = run_mod._resolve_backend(_args("auto"))
    assert image_size == _IMAGE_SIZE


def test_auto_picks_newt_when_url_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NT_INFERENCE_URL", raising=False)
    monkeypatch.setenv("NT_API_KEY", "nt_real_key_for_test")

    image_size, model = run_mod._resolve_backend(_args("auto"))
    assert image_size == _NEWT_IMAGE_SIZE
    assert model == "so101"


def test_modal_sets_the_placeholder_key_automatically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Users should never have to know about the dummy-key quirk."""
    import os

    monkeypatch.setenv("NT_INFERENCE_URL", URL)
    monkeypatch.delenv("NT_API_KEY", raising=False)

    run_mod._resolve_backend(_args("modal"))
    assert os.environ["NT_API_KEY"] == "dummy"


def test_modal_without_a_url_fails_with_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NT_INFERENCE_URL", raising=False)
    with pytest.raises(SystemExit) as exc:
        run_mod._resolve_backend(_args("modal"))
    assert exc.value.code == 2


def test_newt_clears_a_leftover_inference_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hazard this function exists for."""
    import os

    monkeypatch.setenv("NT_INFERENCE_URL", URL)
    monkeypatch.setenv("NT_API_KEY", "nt_real_key_for_test")

    image_size, _ = run_mod._resolve_backend(_args("newt"))
    assert "NT_INFERENCE_URL" not in os.environ
    assert image_size == _NEWT_IMAGE_SIZE


def test_newt_rejects_the_dummy_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 'dummy' left over from a Modal session must not reach the registry."""
    monkeypatch.delenv("NT_INFERENCE_URL", raising=False)
    monkeypatch.setenv("NT_API_KEY", "dummy")

    with pytest.raises(SystemExit) as exc:
        run_mod._resolve_backend(_args("newt"))
    assert exc.value.code == 2


def test_newt_passes_the_model_tag_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NT_INFERENCE_URL", raising=False)
    monkeypatch.setenv("NT_API_KEY", "nt_real_key_for_test")

    _, model = run_mod._resolve_backend(_args("newt", model="my-custom-tag"))
    assert model == "my-custom-tag"


def test_backend_image_sizes_disagree() -> None:
    """If these ever converge, the whole per-backend plumbing can be deleted."""
    assert _IMAGE_SIZE != _NEWT_IMAGE_SIZE
