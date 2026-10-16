"""Command-line entry point for *4,198 Feet*."""

from __future__ import annotations

import argparse
import json

from data import run_phase1
from phase1_checks import run_phase1_checks
from phase1_managed import run_phase1_managed
from phase1_decision import run_managed_decision
from phase1_aoi import run_managed_aoi
from phase2 import run_phase2
from phase3 import run_phase3
from mesh_density import build_density_comparison
from relief import build_relief
from film import render_final_film, render_phase4_preview


def main() -> None:
    """Run one approved project phase at a time."""
    parser = argparse.ArgumentParser(description="Build 4,198 Feet")
    parser.add_argument(
        "phase",
        choices=(
            "phase1",
            "phase1-checks",
            "phase1-managed",
            "phase1-decision",
            "phase1-aoi",
            "phase2",
            "phase2-density",
            "phase2-relief",
            "phase3",
            "phase4-preview",
            "phase4-final",
        ),
    )
    parser.add_argument(
        "--terrain",
        action="store_true",
        help="Enable opt-in 3D DEM geometry for Phase 4 renders",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=2.0,
        help="Vertical terrain exaggeration (default: 2.0)",
    )
    args = parser.parse_args()
    if args.phase == "phase1":
        print(json.dumps(run_phase1(), indent=2))
    elif args.phase == "phase1-checks":
        print(json.dumps(run_phase1_checks(), indent=2))
    elif args.phase == "phase1-managed":
        print(json.dumps(run_phase1_managed(), indent=2))
    elif args.phase == "phase1-decision":
        print(json.dumps(run_managed_decision(), indent=2))
    elif args.phase == "phase1-aoi":
        print(json.dumps(run_managed_aoi(), indent=2))
    elif args.phase == "phase2":
        print(json.dumps(run_phase2(), indent=2))
    elif args.phase == "phase2-density":
        print(json.dumps(build_density_comparison(), indent=2))
    elif args.phase == "phase2-relief":
        print(json.dumps(build_relief(), indent=2))
    elif args.phase == "phase3":
        print(json.dumps(run_phase3(), indent=2))
    elif args.phase == "phase4-preview":
        print(
            json.dumps(
                render_phase4_preview(
                    terrain=args.terrain,
                    exaggeration=args.exaggeration,
                ),
                indent=2,
            )
        )
    elif args.phase == "phase4-final":
        print(
            json.dumps(
                render_final_film(
                    terrain=args.terrain,
                    exaggeration=args.exaggeration,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
