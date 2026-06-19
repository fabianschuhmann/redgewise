from __future__ import annotations
import os
import argparse
from pathlib import Path

from redgewise import __version__
from redgewise.build import run_build

from redgewise.plot_vmd import run_plot_vmd


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


    plot = subparsers.add_parser(
        "plot",
        help="Plot or export analyses from a redgewise build output.",
    )

    plot_subparsers = plot.add_subparsers(dest="plot_command", required=True)

    vmd = plot_subparsers.add_parser(
        "vmd",
        help="Write a VMD TCL drawer and pseudo-bead PDB from a redgewise output.",
    )

    vmd.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Redgewise build output directory.",
    )

    vmd.add_argument(
        "-s",
        "--structure",
        type=Path,
        required=True,
        help="Structure file used to compute vertex centers, e.g. .gro, .pdb, or .tpr.",
    )

    vmd.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory for drawer.tcl and network_beads.pdb.",
    )

    vmd.add_argument(
        "--value",
        choices=("vdw", "cl", "vdw+cl", "dvdw", "dcl", "dvdw+dcl"),
        default="vdw+cl",
        help="Edge value used for cylinder radius. Default: vdw+cl.",
    )

    vmd.add_argument(
        "--radius-scale",
        type=float,
        default=0.01,
        help="Cylinder radius scaling factor applied to abs(mean edge value).",
    )

    vmd.add_argument(
        "--min-abs-value",
        type=float,
        default=0.0,
        help="Only draw edges with abs(mean selected value) >= this threshold.",
    )

    vmd.add_argument(
        "--exclude-resname",
        action="append",
        default=[],
        metavar="RESNAME",
        help="Exclude vertices with this residue_name. Can be used multiple times.",
    )

    vmd.add_argument(
        "--max-edges",
        type=int,
        default=None,
        help="Optionally draw only the strongest N edges by abs(value).",
    )
    vmd.add_argument(
        "--coordinate-unit",
        choices=("auto", "angstrom", "nm"),
        default="auto",
        help=(
            "Coordinate unit returned by the structure reader. "
            "PDB/TCL output is always written in Angstrom. "
            "Use 'nm' if the pseudo-bead structure appears 10x too small."
        ),
    )

    vmd.set_defaults(func=run_plot_vmd)

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