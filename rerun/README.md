# rerun_lama

Hackathon project: analyzing xylophone-playing robot arm data from an SO-100 robot arm.

Data is in LeRobot v3 dataset format (parquet) and/or already logged to Rerun as `.rrd`
recording files. There is no audio in the dataset — all analysis (strike detection,
timing, velocity/acceleration metrics, etc.) is derived from joint kinematics.

## Layout

- `data/` — drop `.rrd` recordings and/or the LeRobot parquet dataset here (gitignored)
- `scripts/` — analysis scripts (strike detection, metrics, etc.)
- `notebooks/` — exploratory Jupyter notebooks
- `output/` — derived data, exported queries, generated Rerun blueprints, demo assets (gitignored)

## Setup

All commands below assume you're inside this `rerun/` directory.

```
cd rerun
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux/Git Bash
pip install -r requirements.txt
```

## Usage

```
python scripts/inspect_data.py [path/to/file.rrd]
```

With no argument, the script scans `data/` for `.rrd` files and inspects each one found.
