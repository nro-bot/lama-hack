"""
Modal-hosted policy server speaking newt's msgpack-WebSocket protocol.

The newt hosted API resolves `model=` against its own registry and has no way to
point at an arbitrary HuggingFace checkpoint. So instead of registering the
fine-tune with New Theory, we serve it ourselves and set NT_INFERENCE_URL on the
client, which bypasses registry discovery entirely (robot.py:821) and points the
WebSocket here.

Deploy:
    modal deploy server/modal_ws.py

Then, on the robot machine:
    export NT_INFERENCE_URL=wss://<workspace>--xylophone-policy-policy-web.modal.run
    export NT_API_KEY=dummy        # non-empty or the SDK raises AuthError first

Protocol (see server/codec.py for the wire format):
    in   {"type": "obs", "prompt": str, "state": (6,), "images": {top, side}}
    out  {"type": "action", "chunk": (N, 6) float32}
"""
from __future__ import annotations

import os
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Checkpoint. Override with MOLMOACT_CHECKPOINT at deploy time.
#
# This is a LeRobot-saved policy directory: finetune_modal.py trains via
# `lerobot_train --policy.path=...`, whose output is LeRobot's config.json +
# model.safetensors schema, NOT the raw transformers layout that the
# allenai/MolmoAct2-SO100_101 card documents. That distinction decides which
# loader works — see _load_policy.
# ---------------------------------------------------------------------------

CHECKPOINT = os.environ.get(
    "MOLMOACT_CHECKPOINT", "ArjunPrasaath/play_xylophone_100"
)

# From the checkpoint's config.json — the client must agree with these or the
# model silently sees the wrong thing:
#   input_features  cam0/cam1 [3, 224, 224], state [6]     (embodiment._IMAGE_SIZE)
#   chunk_size / n_action_steps  30                        (MAX_ACTIONS_PER_CHUNK)
#   normalization   VISUAL=IDENTITY, STATE/ACTION=QUANTILES
#
# QUANTILES is why the preprocessor is mandatory rather than nice-to-have: the
# quantile statistics live in policy_preprocessor_*.safetensors, and without
# them the policy receives raw joint values where it expects normalized ones.
EXPECTED_IMAGE_SIZE = 224

# Camera rename, and it must match finetune_modal.py's --rename_map exactly:
#   observation.images.top  -> observation.images.cam0
#   observation.images.side -> observation.images.cam1
# Get this backwards and the model sees the side view where it expects the top
# one, which looks like a badly-trained policy rather than a wiring bug.
_CAMERA_RENAME = {"top": "cam0", "side": "cam1"}

app = modal.App("xylophone-policy")

# The image must match the one the fine-tune was TRAINED in
# (so100-hackathon/tools/apps/finetune_modal.py:62-71). Two details are
# load-bearing and both were wrong on the first deploy:
#
#   - Python 3.12, not 3.11: lerobot>=0.5 requires it.
#   - lerobot[molmoact2] from git main, not PyPI. The released wheel has no
#     lerobot.policies.molmoact2 at all, so MolmoAct2Policy simply doesn't
#     exist and the loader fails at import.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")  # pip needs git to install lerobot from a git ref
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "msgpack",
        "numpy",
        "pillow",
        "fastapi[standard]",
        # We run uvicorn ourselves (see @modal.web_server); websockets is the
        # protocol implementation it needs for the WS upgrade.
        "uvicorn[standard]",
        "websockets",
        "lerobot[molmoact2] @ git+https://github.com/huggingface/lerobot.git@main",
    )
    .env({"HF_HOME": "/cache"})
    .add_local_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "codec.py"),
        "/root/codec.py",
    )
)

cache_volume = modal.Volume.from_name("molmoact2-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/cache": cache_volume},
    # No HF secret: ArjunPrasaath/play_xylophone_100 is public. If you later
    # point CHECKPOINT at a private repo, add
    #   secrets=[modal.Secret.from_name("huggingface-secret")]
    # with HF_TOKEN set.
    scaledown_window=900,
    timeout=3600,
    # One container, kept warm. Both halves matter:
    #
    #   max_containers=1  — without it every concurrent connection attempt spun
    #     up its own container, each independently downloading the full
    #     checkpoint. The logs showed several "Fetching 18 files" races.
    #   min_containers=1  — a cold start is ~2-4 min (download + 1295 weight
    #     shards + GPU transfer), far longer than any sane client timeout. The
    #     arm should never wait on that.
    #
    # This holds a GPU allocated and therefore costs money continuously. Drop
    # min_containers back to 0 when you're done demoing.
    min_containers=1,
    max_containers=1,
)
class Policy:
    """Loads the fine-tune once per container, serves it over a WebSocket."""

    @modal.enter()
    def load(self) -> None:
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy, self.preprocessor, self.postprocessor = _load_policy(
            CHECKPOINT, self.device
        )

        # Persist the HuggingFace download to the volume. Without this commit
        # the weights land in the container's ephemeral copy of /cache and the
        # next cold start re-downloads all 18 files (~80s) from scratch.
        try:
            cache_volume.commit()
        except Exception as exc:
            print(f"[policy] volume commit failed, cold starts stay slow: {exc}")

        print(f"[policy] ready: {CHECKPOINT} on {self.device}")

    # -----------------------------------------------------------------------

    def _infer(self, frame: dict):
        """One obs frame -> one (N, 6) float32 action chunk."""
        import numpy as np
        from codec import get_key, get_str

        torch = self.torch
        prompt = get_str(frame, "prompt")
        state = np.asarray(get_key(frame, "state"), dtype=np.float32)
        images = get_key(frame, "images") or {}

        # Images arrive CHW uint8, already resized to EXPECTED_IMAGE_SIZE by the
        # client (embodiment.read_state), so don't resize again — just scale to
        # [0,1] float, which is LeRobot's image convention. config.json has
        # VISUAL: IDENTITY, so the pipeline passes them through unchanged from
        # there and the model's own processor handles the rest.
        batch = {"task": [prompt]}
        for cam, renamed in _CAMERA_RENAME.items():
            arr = np.asarray(get_key(images, cam), dtype=np.float32) / 255.0
            batch[f"observation.images.{renamed}"] = (
                torch.from_numpy(arr).unsqueeze(0).to(self.device)
            )
        batch["observation.state"] = (
            torch.from_numpy(state).unsqueeze(0).to(self.device)
        )

        # Both processors are always present — _load_policy raises rather than
        # serve without the QUANTILES normalization statistics.
        with torch.inference_mode():
            batch = self.preprocessor(batch)
            chunk = self.policy.predict_action_chunk(batch)
            chunk = self.postprocessor(chunk)

        chunk = chunk[0] if chunk.ndim == 3 else chunk
        return chunk.float().cpu().numpy()

    # -----------------------------------------------------------------------

    # @modal.web_server + the plain `websockets` library. No FastAPI, no
    # uvicorn, no ASGI anywhere. That is deliberate, after two distinct
    # in-production failures that only exist inside Modal's plumbing:
    #
    #   1. @modal.asgi_app: Modal serializes every ASGI message itself, and its
    #      protobuf for websocket.close rejects reason=None — which Starlette
    #      sends by default (modal/_serialization.py:248). Every connection died
    #      at the handshake with HTTP 500 "bad argument type for built-in
    #      operation", in Modal's bridge, below our code.
    #   2. @modal.web_server + uvicorn/FastAPI: uvicorn answered every proxied
    #      websocket upgrade with 403 (app-close-before-accept) in the
    #      container, while the identical app + identical forwarded headers
    #      pass locally. Cause never isolated; the stack was removed instead.
    #
    # The websockets library is also what the newt client itself uses, so both
    # ends of the wire now speak the same, well-trodden implementation. /health
    # is answered from process_request: any request that is not a websocket
    # upgrade gets a plain HTTP 200.
    @modal.web_server(port=8000, startup_timeout=900)
    def web(self) -> None:
        import asyncio
        import threading

        def _run() -> None:
            asyncio.run(self._serve_forever())

        # Daemon thread: web_server expects this method to return once the
        # port is listening, not to block serving.
        threading.Thread(target=_run, daemon=True).start()

    async def _serve_forever(self) -> None:
        import asyncio
        import http

        from websockets.asyncio.server import serve

        def process_request(connection, request):
            # Non-upgrade requests (health checks, readiness probes, curious
            # browsers) get an HTTP answer instead of a failed WS handshake.
            if "upgrade" not in (request.headers.get("Connection") or "").lower():
                return connection.respond(
                    http.HTTPStatus.OK,
                    f'{{"status": "ok", "checkpoint": "{CHECKPOINT}"}}\n',
                )
            return None  # continue with the websocket handshake, any path

        async with serve(
            self._session,
            host="0.0.0.0",
            port=8000,
            process_request=process_request,
            # Obs frames are ~300 KB (2 cams @ 224px); leave generous headroom
            # rather than inherit the library's 1 MiB default as a cliff.
            max_size=16 * 1024 * 1024,
        ):
            await asyncio.Future()  # serve until the container is torn down

    async def _session(self, ws) -> None:
        from websockets.exceptions import ConnectionClosed

        from codec import get_key, get_str, pack, unpack

        started = time.monotonic()
        max_duration: float | None = None
        frames = 0

        try:
            while True:
                raw = await ws.recv()
                if isinstance(raw, str):
                    continue  # protocol is binary msgpack; ignore stray text
                frame = unpack(raw)
                if get_str(frame, "type") != "obs":
                    continue

                # max_duration arrives on the first frame only. We terminate
                # client-side (the sequencer raises SequenceComplete), so this
                # is purely a backstop against a wedged client leaving the arm
                # running.
                if max_duration is None:
                    max_duration = get_key(frame, "max_duration")
                    print(
                        f"[policy] session start: prompt="
                        f"{get_str(frame, 'prompt')!r} max_duration={max_duration}"
                    )

                if max_duration and time.monotonic() - started > float(max_duration):
                    await ws.send(
                        pack({"type": "terminal", "stop_reason": "max_duration"})
                    )
                    await ws.close(code=1000, reason="max_duration")
                    return

                chunk = self._infer(frame)
                frames += 1
                await ws.send(pack({"type": "action", "chunk": chunk}))

        except ConnectionClosed:
            print(f"[policy] client disconnected after {frames} frames")
        except Exception as exc:
            # Surface the failure instead of leaving the arm hanging on a recv
            # that will never complete.
            import traceback

            traceback.print_exc()
            try:
                await ws.send(
                    pack(
                        {
                            "type": "terminal",
                            "stop_reason": f"server_error: {type(exc).__name__}: {exc}",
                        }
                    )
                )
            except Exception:
                pass
            try:
                await ws.close(code=1011, reason=f"server_error: {type(exc).__name__}")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_policy(checkpoint: str, device: str):
    """Load the fine-tune. Returns (policy, preprocessor, postprocessor).

    LeRobot only, no fallback. play_xylophone_100 ships config.json,
    model.safetensors and policy_{pre,post}processor*.safetensors — LeRobot's
    schema, the output of `lerobot_train --policy.path`. Its config.json has no
    `model_type` key at all, so the transformers loader cannot read it even in
    principle.

    An earlier version fell back to transformers when this import failed. That
    was actively harmful: the first deploy failed on a one-line cause
    (`No module named lerobot.policies.molmoact2` — the PyPI wheel has no
    molmoact2, you need the [molmoact2] extra from git main), and the fallback
    buried it under a 200-line transformers traceback about `model_type`. Fail
    on the real error instead.
    """
    try:
        from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    except ImportError as exc:
        raise ImportError(
            f"lerobot has no molmoact2 support ({exc}).\n"
            "The image must install lerobot[molmoact2] from git main — the "
            "released PyPI wheel does not ship this policy. Match the image the "
            "fine-tune was trained in (finetune_modal.py:62-71)."
        ) from exc

    policy = MolmoAct2Policy.from_pretrained(checkpoint).to(device).eval()

    # The normalization statistics. config.json declares STATE and ACTION as
    # QUANTILES, so these are REQUIRED: without them the policy receives raw
    # joint values where it expects normalized ones and returns actions in the
    # wrong units — at the servos, a fast move to a garbage pose. Let this raise.
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=checkpoint
    )

    # Guard the client/server contract that has no other checkpoint: the
    # embodiment resizes frames client-side, so a mismatch here is invisible
    # until the model produces subtly wrong actions.
    declared = getattr(policy.config, "input_features", {})
    for cam in ("cam0", "cam1"):
        feature = declared.get(f"observation.images.{cam}")
        shape = getattr(feature, "shape", None) if feature is not None else None
        if shape and tuple(shape)[-1] != EXPECTED_IMAGE_SIZE:
            print(
                f"[policy] WARNING: {cam} expects {tuple(shape)} but this server "
                f"assumes {EXPECTED_IMAGE_SIZE}px. Update embodiment._IMAGE_SIZE "
                "and EXPECTED_IMAGE_SIZE to match."
            )

    return policy, preprocessor, postprocessor
