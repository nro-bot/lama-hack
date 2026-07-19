"""Static checks on server/modal_ws.py that don't require torch, a GPU, or Modal.

The server can only be exercised for real by deploying it, and a deploy plus a
cold start is minutes. So the cheap, mechanical mistakes should be caught here
instead of in production logs.

Written after exactly that: `_load_policy` was changed from returning 4 values to
3, the caller's unpack was updated, but one `return` statement still carried the
old arity. It deployed fine and failed at container start with "too many values
to unpack". Pyflakes does not catch this; this does.

Everything here is AST-level — the module is parsed, never imported.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parent.parent / "server" / "modal_ws.py"


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(SERVER.read_text())


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in {SERVER.name}")


def _return_arities(func: ast.FunctionDef) -> list[int]:
    """Element count of every non-bare `return` directly inside `func`.

    Skips returns nested in inner function definitions (the ASGI route handlers
    live inside web(), and they are not what we're checking).
    """
    arities = []
    for node in ast.walk(func):
        if isinstance(node, ast.FunctionDef) and node is not func:
            continue
        if isinstance(node, ast.Return) and node.value is not None:
            if isinstance(node.value, ast.Tuple):
                arities.append(len(node.value.elts))
            else:
                arities.append(1)
    return arities


def test_load_policy_returns_a_consistent_arity(tree: ast.Module) -> None:
    """Every return path out of _load_policy must agree with the others."""
    arities = set(_return_arities(_find_function(tree, "_load_policy")))
    assert len(arities) == 1, (
        f"_load_policy has return statements of differing arity: {sorted(arities)}. "
        "Every path must return the same number of values."
    )


def test_the_caller_unpacks_what_load_policy_returns(tree: ast.Module) -> None:
    """The bug this file exists for."""
    returned = set(_return_arities(_find_function(tree, "_load_policy")))
    assert len(returned) == 1
    arity = returned.pop()

    load = _find_function(tree, "load")
    unpacks = [
        len(node.targets[0].elts)
        for node in ast.walk(load)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Tuple)
        and "_load_policy" in ast.dump(node.value)
    ]
    assert unpacks, "load() no longer calls _load_policy via tuple unpacking"
    for got in unpacks:
        assert got == arity, (
            f"load() unpacks {got} values but _load_policy returns {arity}"
        )


def test_no_stale_flavor_references(tree: ast.Module) -> None:
    """The transformers fallback is gone; its plumbing should be too."""
    source = SERVER.read_text()
    assert "self.flavor" not in source
    assert "_nt_processor" not in source


def test_the_camera_rename_is_still_top_side_to_cam0_cam1() -> None:
    """Reversing this looks like a bad policy rather than a wiring bug."""
    import sys

    sys.path.insert(0, str(SERVER.parent))
    source = SERVER.read_text()
    assert '"top": "cam0"' in source
    assert '"side": "cam1"' in source


def test_the_image_installs_lerobot_with_the_molmoact2_extra() -> None:
    """The PyPI wheel has no molmoact2 module; it must come from git main.

    This was the first deploy's failure: `No module named
    lerobot.policies.molmoact2`. The image has to match the one the fine-tune
    was trained in (finetune_modal.py:62-71).
    """
    source = SERVER.read_text()
    assert "lerobot[molmoact2]" in source, "lerobot must be installed with the molmoact2 extra"
    assert "git+https://github.com/huggingface/lerobot.git@main" in source
    assert 'python_version="3.12"' in source, "lerobot>=0.5 requires Python 3.12"
