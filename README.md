# Reasmory

Spatial reasoning over images and video by **reconstructing the scene in 3D and
querying it with tools**, instead of asking a VLM to infer geometry from pixels
alone.

A question is answered in stages:

| Stage | Role |
| --- | --- |
| **Decomposer** | Turns the question into a structured form: where to stand, how to face, what is being asked, and in whose frame of reference. |
| **Planner** | Emits a short program that composes tool primitives (query object positions, set/move the viewpoint, render an egocentric view or a bird's-eye view). |
| **Executor** | Verifies the plan, then runs it against a 3D reconstruction of the scene. |
| **Reasoner** | Answers the original question from the rendered observations, each captioned with how it was produced and where to look. |

Static scenes are reconstructed with Pi3 (or metric Pi3x for measurement
questions); dynamic video is reconstructed with Flow3R. Object instances are
grounded with SAM3 and merged across frames — spatially for static scenes,
temporally for video.

## Setup

```bash
pip install -r requirements.txt          # see "Requirements" below
cp .env.example .env
$EDITOR .env                             # API keys + data/cache roots
set -a && . ./.env && set +a
```

API keys are read from the environment and are never hardcoded; a missing key
raises an explicit error rather than falling back to a baked-in credential.

Point `REASMORY_DATA_ROOT` at your copy of the benchmark media. The shipped
annotation files contain the absolute media paths of the machine that generated
them, and are transparently re-rooted at load time (`tools/paths.py`) — you do not
need to rewrite any JSON.

The MindCube image subset ships with the repository
(`evaluation/annotation_assets/`), so `mindcube_tiny_no_text_shuffled` runs
straight after cloning. The VSI-Bench and VLM4D video benchmarks reference large
video files that are not redistributed here: download them from their original
releases and point `REASMORY_DATA_ROOT` at the parent directory, keeping the
layout the annotations expect (`VSIBench/...`, `VLM4D/videos_real/...`).

| Variable | Purpose |
| --- | --- |
| `REASMORY_DATA_ROOT` | Benchmark videos and image folders |
| `REASMORY_WORKSPACE_ROOT` | Scratch space for rendered tool artifacts |
| `REASMORY_SPATIAL_MEMORY_CACHE` | Precomputed static (Pi3) reconstructions |
| `REASMORY_SPATIAL_MEMORY_CACHE_PI3X` | Precomputed metric (Pi3x) reconstructions |
| `REASMORY_DYNAMIC_MEMORY_CACHE` | Precomputed dynamic (Flow3R) reconstructions |
| `REASMORY_SAM3_ROOT` | SAM3 checkout (defaults to a `third_party/sam3` sibling) |

Reconstruction is expensive, so precompute the caches once:

```bash
python evaluation/precompute_vsibench_tiny_spatial_memory_cache.py   # static
python evaluation/precompute_vlm4d_dynamic_memory_cache.py           # dynamic
```

## Running an evaluation

`evaluation/run_eval.py` is the single entry point for both scene types and picks
the backend from the benchmark name — static (Pi3/Pi3x) or dynamic (Flow3R):

```bash
python evaluation/run_eval.py --list-benchmarks     # available benchmarks + scene type

# static scenes (MindCube, VSI-Bench)
python evaluation/run_eval.py --benchmark mindcube_tiny_no_text_shuffled --model_type my_run \
    --planner_model_path gemini-3-flash-preview \
    --reasoner_model_path gemini-3-flash-preview \
    --import_existing_plan <planner_results.json> --bev_fallback

# dynamic scenes (VLM4D)
python evaluation/run_eval.py --benchmark vlm4d_ego_50 --model_type my_run
```

All remaining arguments are forwarded to the selected backend; use
`--benchmark <name> --help` for that backend's options and `--scene` to override
the inferred routing. Results land in `eval_results/eval_<benchmark>/`.

### Wrapper scripts

`scripts/` wraps the common workflows. Each reads its configuration from the
environment (`scripts/common.sh` holds the shared defaults and loads `.env`):

```bash
scripts/precompute_cache.sh static  vsibench_tiny      # build reconstruction caches
scripts/precompute_cache.sh dynamic vlm4d_ego_50

scripts/run_pipeline.sh mindcube_tiny_no_text_shuffled                  # decomposer -> planner -> reasoner
scripts/run_pipeline.sh vlm4d_ego_50 vlm4d_allo_50     # dynamic scenes
REASONER_MODEL=gpt-5-mini MAX_SAMPLES=20 scripts/run_pipeline.sh vsibench_tiny

scripts/run_baseline.sh mindcube_tiny_no_text_shuffled                  # direct-VLM reference point
BASELINE_MODEL=gpt-5-mini scripts/run_baseline.sh vsibench_tiny

scripts/run_stage.sh decomposer mindcube_tiny_no_text_shuffled          # iterate on a single stage
scripts/run_stage.sh planner    mindcube_tiny_no_text_shuffled
```

`run_pipeline.sh` reuses a stage's cached results when present; set
`FORCE_STAGES=1` to recompute.

## Repository layout

### `tools/` — the pipeline
- `paths.py` — environment-driven filesystem layout
- `llm_cfg.py` — provider configuration, keys from the environment
- `spatial_memory.py` — 3D scene memory: reconstruction, viewpoints, rendering
- `agent_tools.py` — the tool primitives the planner composes
- `prompt.py` — decomposer / planner / reasoner prompts
- `plan_verifier.py`, `plan_compiler.py`, `plan_executor.py`, `plan_rules/` — the
  plan DSL: verification, optimisation and execution
- `plan_auto_repair.py` — repairs plans that fail verification
- `reasoner_prompt_builder.py` — builds the reasoner context and per-observation
  captions ("how this view was produced, and where to look in it")
- `vis_utils.py` — semantic and RGB bird's-eye-view rendering
- `vlm4d_*.py` — dynamic-video branch: camera ego-motion analysis, temporal
  instance merging, per-archetype evidence and prompts

### `evaluation/` — harness
- `run_eval.py` — unified entry point
- `eval_three_step.py` — static-scene pipeline
- `eval_vlm4d_three_step.py` — dynamic-scene pipeline
- `eval_agent.py` — direct-VLM baseline
- `eval_decomposer.py`, `eval_dsl_planner.py` — per-stage evaluation
- `annotation/` — benchmark annotation files
- `precompute_*.py` — reconstruction cache builders

## Requirements

```
torch==2.6.0
transformers
qwen-agent
numpy, scipy, opencv-python, pillow, matplotlib
ray, tqdm
```

External components, installed separately:
- **Pi3 / Pi3x** — static scene reconstruction
- **Flow3R** — dynamic video reconstruction (run in an isolated environment)
- **SAM3** — open-vocabulary segmentation for object grounding
