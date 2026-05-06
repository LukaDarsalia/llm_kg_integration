"""CLI: `python -m llm_kg run path/to/config.yaml`."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from llm_kg.runner.experiment import Experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-kg")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run an experiment from a YAML config")
    run_p.add_argument("config", type=Path)

    args = parser.parse_args(argv)

    if args.cmd == "run":
        metrics = asyncio.run(Experiment(args.config).run())
        console = Console()
        table = Table(title=f"Results: {args.config}")
        table.add_column("metric")
        table.add_column("value", justify="right")
        for k, v in sorted(metrics.items()):
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        console.print(table)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
