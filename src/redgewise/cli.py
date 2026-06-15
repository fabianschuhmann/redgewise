from __future__ import annotations
import os
import argparse
from pathlib import Path

from redgewise import __version__
from redgewise.build import run_build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="redgewise",
        description="Residue Energy edge-wise analysis from GROMACS simulations.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"redgewise {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="Build a residue-level energy network.",
    )
    build.add_argument("-s", "--tpr", type=Path, required=True)
    build.add_argument("-t", "--trajectory", type=Path, required=True)
    build.add_argument("-p", "--topology", type=Path,required=True)
    build.add_argument("-o", "--output", type=Path, required=True)
    build.add_argument("-f", "--mdp",type=Path,required=True)
    build.add_argument(
        "--Worker",
        "--workers",
        dest="workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers. Default: number of available CPU cores.",
    )

    build.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU backend if available. Default: CPU only.",
    )

    build.add_argument(
        "--high_res",
        action="append",
        default=[],
        metavar="RESNAME",
        help=(
            "Residue name to keep at atom resolution instead of reducing to residue "
            "level. Can be used multiple times."
        ),
    )

    build.add_argument(
        "--low_res",
        action="append",
        default=[],
        metavar="RESNAME",
        help=(
            "Residue name for which all residues of that type are grouped into one "
            "interaction group. Can be used multiple times."
        ),
    )

    build.add_argument(
        "--bundle",
        action="append",
        nargs="+",
        default=[],
        metavar="RESNAME",
        help=(
            "Bundle multiple residue names into one interaction group. Can be used "
            "multiple times. Example: --bundle POPC POPE DOPC"
        ),
    )
    build.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Analyze every Nth trajectory frame. Default: 1.",
    )
    build.add_argument(
        "--frames-per-part",
        type=int,
        default=10,
        help="Number of processed frames written per Parquet part. Default: 10.",
    )


    build.set_defaults(func=run_build)

    info = subparsers.add_parser(
        "info",
        help="Show program information.",
    )
    info.set_defaults(func=run_info)

    args = parser.parse_args(argv)
    return args.func(args)


def run_info(args: argparse.Namespace) -> int:
    print(f"redgewise {__version__}")
    print("REdgEwise: Residue Energy edge-wise analysis")
    return 0