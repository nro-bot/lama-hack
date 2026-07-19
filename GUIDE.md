# End-to-end runbook

How to take this repo from a fresh clone to an SO-101 arm playing notes, with
**your own fine-tuned MolmoAct2 checkpoint from HuggingFace**. Written for
someone who has never seen this codebase. Every failure mode listed here
actually happened; the fixes are the ones that worked.

Total time, assuming the hardware exists and your checkpoint is trained:
~30 minutes, most of it Modal's first image build.

---

## 0. What you need before starting

- **A fine-tuned checkpoint on HuggingFace, in LeRobot format.** It is LeRobot
  format if the repo files include `policy_preprocessor.json`,
  `policy_postprocessor.json` and `*_normalizer.safetensors` alongside
  `config.json` + `model.safetensors`. That is what
  `lerobot_train --policy.path=...` produces (see
  `so100-hackathon/tools/apps/finetune_modal.py`). A raw-transformers MolmoAct2
  checkpoint will NOT load — this server is LeRobot-only, on purpose.
- **A Modal account**, authenticated on this machine (`uv run modal token new`
  if `~/.modal.toml` doesn't exist).
- **The SO-101 arm + two USB webcams** plugged into this machine.
- **uv** installed (`brew install uv`).
- Optional: an `ANTHROPIC_API_KEY`, only for free-text songs
  ("play twinkle twinkle"). Explicit note lists ("C D E") never need it.

```bash
git clone https://github.com/nro-bot/lama-hack.git && cd lama-hack
uv sync --extra hardware --extra server
```

> If you skip `--extra hardware` you'll hit
> `ModuleNotFoundError: No module named 'scservo_sdk'` the moment the arm
> connects. Both extras together is always safe.

---

## 1. Point the code at YOUR checkpoint

Three places know about the model. Check all three.

### 1a. The server (`server/modal_ws.py`)

Change the default, or override at deploy time without editing anything:

```bash
export MOLMOACT_CHECKPOINT="your-hf-user/your_model_name"
```

If your repo is **private**, also create a Modal secret and re-enable it in the
`@app.cls(...)` block (there's a comment showing exactly what to add):

```bash
uv run modal secret create huggingface-secret HF_TOKEN=hf_xxxx
```

### 1b. The client constants must match your checkpoint's config.json

Download and read it (30 seconds, saves an hour of confusing arm behavior):

```bash
uv run --with huggingface_hub python -c "
from huggingface_hub import hf_hub_download; import json
c = json.load(open(hf_hub_download('your-hf-user/your_model_name', 'config.json')))
print('image shape :', c['input_features']['observation.images.cam0']['shape'])
print('chunk_size  :', c['chunk_size'])
print('norm        :', c['normalization_mapping'])
print('cameras     :', [k for k in c['input_features'] if 'images' in k])
"
```

Then reconcile:

| config.json says | must match | currently |
|---|---|---|
| image shape `[3, H, H]` | `_IMAGE_SIZE` in `embodiment.py` | 224 |
| `chunk_size` | `MAX_ACTIONS_PER_CHUNK` ≤ it, in `embodiment.py` | 15 (of 30) |
| camera feature names | `_CAMERA_RENAME` in `server/modal_ws.py` (`top→cam0`, `side→cam1`) | cam0/cam1 |

Finally update `CHECKPOINT` in `tests/test_contract.py` to your repo id and run:

```bash
uv run --with pytest --with huggingface_hub python -m pytest tests/test_contract.py -v
```

Green means the client and your checkpoint agree. **These mismatches are
silent at runtime** — the model just quietly sees wrongly-sized images or the
wrong camera in each slot and moves badly, which looks exactly like "my model
is undertrained" when it isn't.

### 1c. The instruction labels (`notes.py`) — the one people forget

`LABELS` must be **byte-for-byte** the instruction strings your dataset was
recorded with. This repo's set is:

```
"Hitting note A (blue)", "Hitting note B (purple)", "Hitting note C (big red)",
"Hitting note C (small red)", "Hitting note D (orange)", "Hitting note E (yellow)",
"Hitting note F (green)", "Hitting note G (blue)"
```

If your dataset used different phrasing — even `"hit note A"` vs
`"Hitting note A (blue)"` — edit `LABELS`. Anything else is an
out-of-distribution instruction and the policy's behavior is undefined. Check
what your dataset actually used:

```bash
# tasks live in the dataset repo's meta/tasks.jsonl (LeRobot v3 layout)
uv run --with huggingface_hub python -c "
from huggingface_hub import hf_hub_download
print(open(hf_hub_download('your-hf-user/your_dataset', 'meta/tasks.jsonl',
      repo_type='dataset')).read())
"
```

After editing `LABELS`, run `uv run --with pytest python -m pytest tests/test_notes.py -q`.

---

## 2. Deploy the policy server on Modal

```bash
uv run modal deploy server/modal_ws.py
```

First deploy builds the image (~5–10 min: torch + lerobot from git). It prints
a URL like:

```
https://<workspace>--xylophone-policy-policy-web.modal.run
```

Wait for the container to load the model, then confirm:

```bash
curl https://<workspace>--xylophone-policy-policy-web.modal.run/health
# -> {"status": "ok", "checkpoint": "your-hf-user/your_model_name"}
```

First health check after a deploy can take 2–4 min (GPU boot + weight load).
Watch progress with `uv run modal app logs xylophone-policy` — you want the
line `[policy] ready: <your checkpoint> on cuda`.

**Do not restructure this server around FastAPI/uvicorn or `@modal.asgi_app`.**
Both were tried; both fail only in production, inside Modal's plumbing (details
in the comment block above `web()` in `server/modal_ws.py`). The plain
`websockets` library under `@modal.web_server` is the configuration that works.

### Cost switch

`min_containers=1` in `server/modal_ws.py` keeps a warm A10G allocated
**continuously** so the demo never waits on a cold start. Set it to `0` and
redeploy the moment you're done for the day.

---

## 3. Smoke-test the server — BEFORE touching the arm

```bash
export NT_INFERENCE_URL=wss://<workspace>--xylophone-policy-policy-web.modal.run   # wss://, not https://
export NT_API_KEY=dummy
uv run --with pytest --with websockets python -m pytest tests/test_fake_client.py -v -s
```

Three tests, one synthetic observation each. Together they prove the msgpack
codec, the camera mapping, the checkpoint loader, and — the important one —
that different instructions produce different action chunks (i.e. the prompt
actually conditions the policy).

> `NT_API_KEY=dummy` is not a placeholder to fill in. The newt SDK refuses to
> construct without *some* key, but with `NT_INFERENCE_URL` set the key is
> never sent anywhere that checks it. Any non-empty string works.

---

## 4. Configure the rig (`~/.config/nt/nt.toml`)

```bash
mkdir -p ~/.config/nt && cp conf/nt.toml.example ~/.config/nt/nt.toml   # if absent
```

Fill in:

- **Arm port** — `ls /dev/tty.usbmodem*` (macOS) or `ls /dev/ttyACM*` (Linux).
  Unplug/replug the arm to see which entry it is.
- **Arm id** — this names the lerobot calibration file, so it must match the
  calibration of **the physical arm plugged in right now**. Convention that
  avoids all confusion: use the USB serial (the `usbmodem` suffix) as the id.
  Wrong id = another arm's joint offsets = systematically wrong poses.
- **Cameras** — `top` looks down at the instrument, `side` looks across it,
  matching how the *training data* was recorded. Indices are integers
  (`index_or_path = 0`), typically 0/1/2 with built-in webcams claiming 0.

If the arm has never been calibrated on this machine:

```bash
uv run lerobot-calibrate --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodemXXXX --robot.id=<your-arm-id>
```

Then verify everything without moving anything:

```bash
uv run python3 run.py --check
uv run python3 run.py --dry-run --song "C D E"    # parse only, no hardware
```

---

## 5. Run

Same shell (or re-export the two env vars):

```bash
uv run python3 run.py --song "C"                       # one note, finger on Ctrl+H
uv run python3 run.py --song "C E G" --seconds-per-note 6   # retargeting test
uv run python3 run.py --song "C D E F G A B C"         # the scale
```

### Choosing the inference backend

Two backends exist; `--backend auto` (the default) picks by environment:

| | `--backend modal` | `--backend newt` |
|---|---|---|
| what runs the policy | your Modal server (§2) | New Theory's hosted registry |
| `NT_INFERENCE_URL` | **required** | must be unset — run.py clears a leftover export for you |
| `NT_API_KEY` | any non-empty value (run.py fills in `dummy`) | a real `nt_...` key, or `uv run newt login` (`~/.nt/credentials`) |
| model | your `MOLMOACT_CHECKPOINT` | registry tag via `--model` (default `so101`) |
| image size sent | 224px (your checkpoint's contract) | 378px (hosted so101 contract) |

```bash
# hosted inference, explicitly:
uv run python3 run.py --backend newt --model so101 --song "C D E"
```

run.py refuses footguns loudly: `--backend newt` with a leftover `dummy` key
errors instead of sending it to the registry, and a stale `NT_INFERENCE_URL`
is cleared for the process rather than silently rerouting the run to Modal.

One caveat on hosted runs: the per-note prompt swap rides the wire identically
on both backends, but whether New Theory's server *honors* a changed prompt
mid-session has never been verified. First hosted run, watch whether note 2
actually retargets; if it doesn't, hosted sequencing needs one-session-per-note
instead — say so and we'll add it.

**Ctrl+H at any time = emergency stop + home.** Ctrl+C also works.

Watch the terminal's `[seq] note N/M: <label>` lines against the arm. The
diagnostic question for a new checkpoint: *does the arm re-aim toward a
different bar at each note boundary?* If yes, the framework and wiring are
fine and everything else is policy quality.

Free text (needs `ANTHROPIC_API_KEY`; always `--dry-run` it first):

```bash
uv run python3 run.py --dry-run --song "play twinkle twinkle little star"
```

---

## 6. Tuning knobs

| symptom | knob | where |
|---|---|---|
| strike stops short / too weak | raise `MAX_ACTIONS_PER_CHUNK` toward `chunk_size` | `embodiment.py` |
| prompt flips mid-swing | raise `--seconds-per-note` (default 4.0; floor ≈ 2.5) | CLI |
| arm idles between notes | lower `--seconds-per-note` | CLI |
| new bar starts without settling | `FIRST_CHUNK_SETTLE_S` (1.5) | `embodiment.py` |
| motion too violent | set `max_relative_target` (start ~15) | `SO101.__init__`, `embodiment.py` |
| bare `C` hits the wrong red bar | `--default-c high`, or say `small C` | CLI |

## 7. Troubleshooting — every one of these actually happened

| error | cause | fix |
|---|---|---|
| `No module named 'scservo_sdk'` | hardware extra not installed | `uv sync --extra hardware --extra server` |
| `could not open port /dev/tty.usbmodem...` | port in nt.toml is stale — arm replugged or swapped | `ls /dev/tty.usbmodem*`, update `port` AND check the `id` matches this arm's calibration |
| `No module named 'lerobot.policies.molmoact2'` (Modal logs) | image installed PyPI lerobot | image must use `lerobot[molmoact2] @ git+...@main` on Python 3.12 — already in `server/modal_ws.py`; don't "simplify" it |
| WS handshake HTTP 500, `bad argument type for built-in operation` | someone reintroduced `@modal.asgi_app` | keep the plain-websockets server |
| WS handshake HTTP 403 | someone reintroduced uvicorn/FastAPI | same |
| `Unrecognized model ... model_type` | checkpoint is LeRobot-format being read by transformers, or vice versa | use a LeRobot checkpoint (§0); the server is LeRobot-only |
| first request hangs 2–4 min | cold start | `min_containers=1` during demo hours; the run.py `connect_timeout` already allows 180s |
| arm moves but aims wrong / "model seems untrained" | contract mismatch (image size, camera swap, wrong labels) | re-do §1b and §1c before blaming the training run |
| every note identical motion | prompt not reaching the model | run the §3 smoke test; `test_different_notes_produce_different_actions` isolates it |

## 8. Repo map

```
run.py           CLI entry — parse song, wire sequencer to newt.Robot
notes.py         text -> trained labels; edit LABELS for a new checkpoint (§1c)
sequencer.py     one WS session, prompt swapped per note (the core trick)
embodiment.py    SO-101 driver; _IMAGE_SIZE + MAX_ACTIONS_PER_CHUNK live here (§1b)
server/
  modal_ws.py    Modal GPU server; CHECKPOINT + camera rename live here (§1a)
  codec.py       msgpack wire format — copied from the newt SDK, do not edit
tests/           all offline except test_fake_client.py (live server) and
                 test_contract.py (reads your checkpoint's config from HF)
```
