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
        ),
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


if __name__ == "__main__":
    main()
