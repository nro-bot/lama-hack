# lama-hack

Play a melody on a xylophone with an SO-101 arm.

The fine-tuned MolmoAct2 policy is single-task: one instruction, one bar struck.
This turns that into a sequencer — you type `C D E F G`, it plays the scale.

```bash
uv run python3 run.py --song "C D E F G"
uv run python3 run.py --song "play twinkle twinkle little star"   # LLM fallback
uv run python3 run.py --dry-run --song "C D E"                    # parse only
```

Press **Ctrl+H** at any time to abort and home the arm.

## How it works

The policy knows eight instructions, and only these eight:

| key | label | bar |
|---|---|---|
| `A` | `Hitting note A (blue)` | blue |
| `B` | `Hitting note B (purple)` | purple |
| `C_LOW` | `Hitting note C (big red)` | big red |
| `C_HIGH` | `Hitting note C (small red)` | small red |
| `D` | `Hitting note D (orange)` | orange |
| `E` | `Hitting note E (yellow)` | yellow |
| `F` | `Hitting note F (green)` | green |
| `G` | `Hitting note G (blue)` | blue |

`notes.py` is the chokepoint that guarantees nothing else ever reaches the
model. It parses your text with a regex, and falls back to Claude only when the
input clearly isn't a note list (`"play twinkle twinkle"`). Even then the model
returns *keys*, constrained to an enum, so it cannot invent an untrained label.

**Retasking without reconnecting.** The interesting part is `sequencer.py`. The
newt SDK rebuilds the prompt from the observation dict on every frame, and an
obs carrying its own `"prompt"` key overrides the one passed to `run()`
(`newt/_client/robot.py:1275`). So the whole song is one `run("")` call, one
WebSocket, with the instruction swapped underneath as notes advance. Reconnecting
per note would work too, but the gap is audible.

**Knowing when a note is done.** There's no success signal on this rig — no
microphone, no force sensor. A note holds for a fixed `--seconds-per-note` and
then advances, which doubles as tempo control. A note is never skipped before the
policy has actually returned a chunk for it, so a slow cold start can't silently
burn through the song.

### Bare "C" is ambiguous

Two red bars, both spelled C. Resolved in three tiers:

1. **Explicit** — `big C`, `small red C`, `C5`, `C'`
2. **Ascending context** — a bare C right after A or B is the octave-completing
   high C, so `C D E F G A B C` plays a real scale
3. **Default** — `--default-c low|high` (low by default)

## Setup

The fine-tune is self-hosted, because New Theory's `model=` is a registry lookup
with no way to point at a HuggingFace checkpoint.

```bash
# 1. deploy the policy server
uv sync --extra server
uv run modal deploy server/modal_ws.py

# 2. point the client at it
export NT_INFERENCE_URL=wss://<workspace>--xylophone-policy-policy-web.modal.run
export NT_API_KEY=dummy      # must be non-empty; the SDK checks it before the URL

# 3. on the robot machine
uv sync --extra hardware
uv run python3 run.py --check
```

`ANTHROPIC_API_KEY` is only needed for the free-text fallback.

## Verifying

Run these in order. Everything up to step 4 works on a laptop with no arm.

```bash
# 1. parser, sequencer, and codec (codec is checked byte-for-byte against the
#    real SDK — a drifted copy fails here, not on the wire mid-performance)
uv run --with pytest --with websockets python -m pytest tests/ -q

# 2. the deployed server: one synthetic obs, real WebSocket, no robot.
#    Do this BEFORE touching the arm — it validates the codec, the cam0/cam1
#    mapping, and whether the checkpoint loaded as LeRobot or fell back to
#    transformers, all in one round-trip.
uv run --with pytest --with websockets python -m pytest tests/test_fake_client.py -v -s

# 3. config and parsing on the rig
uv run python3 run.py --check

# 4. one note, finger on Ctrl+H
uv run python3 run.py --song "C"

# 5. the full scale
uv run python3 run.py --song "C D E F G A B C"
```

## Tuning

| knob | where | note |
|---|---|---|
| `--seconds-per-note` | CLI, default 4.0 | ~2.5s is the floor (1.5s settle + 1.0s strike); below that the prompt flips mid-swing |
| `MAX_ACTIONS_PER_CHUNK` | `embodiment.py`, 15 | of a 30-step chunk. **The knob most likely to need tuning** — too low and the mallet stops mid-swing; raise toward 30 if notes sound weak |
| `FIRST_CHUNK_SETTLE_S` | `embodiment.py`, 1.5 | move-to-start pause at each new bar |
| `max_relative_target` | `SO101.__init__`, `None` | per-step joint clamp; lerobot enforces it natively. Set it once the strike motion is characterized — too tight damps the strike |

### The checkpoint's contract

`ArjunPrasaath/play_xylophone_100` declares, in its `config.json`:

| | value | client constant |
|---|---|---|
| image shape | `[3, 224, 224]` | `embodiment._IMAGE_SIZE` |
| chunk size | 30 | `MAX_ACTIONS_PER_CHUNK` slices this |
| state / action | 6-DOF | `_JOINT_ORDER` |
| normalization | `STATE`/`ACTION` = QUANTILES | processors are **mandatory** server-side |

These are invisible coupling — nothing complains at runtime if they drift, the
model just quietly sees the wrong thing. `tests/test_contract.py` reads the real
config from the Hub and fails if any of them disagree.

## Known failure modes

- **High C lands on the big red bar** — tier-2 disambiguation didn't fire. Say
  `small C` explicitly, or pass `--default-c high`.
- **A and G confused** — both bars are blue. The parser can't be at fault here
  (distinct letters, distinct labels), so this is the model reading color rather
  than position. A data problem, not a code one.
- **A note is silently missed** — expected and not retried. There's no success
  signal to retry on, and stopping the song to re-hit one bar is worse than
  dropping it. Each note boundary is logged, so replay the video against
  `[seq] note N/M:` to see which one.

## Layout

```
run.py          CLI
notes.py        text -> trained labels (no hardware imports)
sequencer.py    the prompt-injection seam
embodiment.py   VENDORED from ../../newt-starter-so101, see its header
server/
  modal_ws.py   Modal policy server
  codec.py      msgpack, copied verbatim from the newt SDK
```
