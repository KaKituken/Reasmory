"""Unified entry point for static-scene and dynamic-scene evaluation.

Static scenes (MindCube, VSI-Bench) and dynamic scenes (VLM4D) run through
different executors -- Pi3/Pi3x reconstruction of a rigid scene versus Flow3R
reconstruction of a moving one -- so they keep separate backends. This script
picks the right one from the benchmark name and forwards every other argument
untouched, so there is a single command to remember:

    python evaluation/run_eval.py --benchmark mindcube_tiny_no_text_shuffled  --model_type run1 ...
    python evaluation/run_eval.py --benchmark vsibench_tiny  --model_type run1 ...
    python evaluation/run_eval.py --benchmark vlm4d_ego_50   --model_type run1 ...

Use `--scene {auto,static,dynamic}` to override the inferred routing, and
`--list-benchmarks` to see what annotations are available.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ANNOTATION_DIR = REPO_ROOT / "evaluation" / "annotation"

# Benchmark-name prefixes handled by the dynamic (moving-scene) backend.
DYNAMIC_PREFIXES = ("vlm4d",)

BACKENDS = {
    "static": "evaluation.eval_three_step",
    "dynamic": "evaluation.eval_vlm4d_three_step",
}


def infer_scene(benchmark: str) -> str:
    name = benchmark.lower()
    return "dynamic" if name.startswith(DYNAMIC_PREFIXES) else "static"


def available_benchmarks() -> list[str]:
    if not ANNOTATION_DIR.is_dir():
        return []
    return sorted(
        p.stem[len("eval_"):]
        for p in ANNOTATION_DIR.glob("eval_*.json")
        if p.stem.startswith("eval_")
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    ap.add_argument("--benchmark", help="Benchmark annotation name (see --list-benchmarks).")
    ap.add_argument("--scene", choices=["auto", "static", "dynamic"], default="auto",
                    help="Override which backend to use. Default: inferred from --benchmark.")
    ap.add_argument("--list-benchmarks", action="store_true",
                    help="Print the available benchmark names and exit.")
    ap.add_argument("-h", "--help", action="store_true", dest="show_help")
    args, passthrough = ap.parse_known_args()

    if args.list_benchmarks:
        names = available_benchmarks()
        for name in names:
            print(f"  {infer_scene(name):8s}  {name}")
        if not names:
            print(f"No annotations found in {ANNOTATION_DIR}")
        return

    if args.show_help and not args.benchmark:
        ap.print_help()
        print("\nAll other arguments are forwarded to the selected backend.")
        print("For backend-specific options run, e.g.:")
        print("  python evaluation/run_eval.py --benchmark mindcube_tiny_no_text_shuffled --help")
        return

    if not args.benchmark:
        ap.error("--benchmark is required (use --list-benchmarks to see the options)")

    scene = infer_scene(args.benchmark) if args.scene == "auto" else args.scene
    module = BACKENDS[scene]

    annotation = ANNOTATION_DIR / f"eval_{args.benchmark}.json"
    if not annotation.exists():
        ap.error(
            f"No annotation file for benchmark '{args.benchmark}' at {annotation}.\n"
            f"Run with --list-benchmarks to see the available names."
        )

    forwarded = ["--benchmark", args.benchmark] + passthrough
    if args.show_help:
        forwarded = ["--help"]

    print(f"[run_eval] benchmark={args.benchmark}  scene={scene}  backend={module}",
          file=sys.stderr)

    # Hand over to the backend as if it had been invoked directly.
    sys.argv = [module.replace(".", "/") + ".py"] + forwarded
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
