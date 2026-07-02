from __future__ import annotations

import argparse
import os
from pathlib import Path

from redgewise import __version__
from redgewise.build import run_build
from redgewise.info import run_info
from redgewise.manual import ManualAction
from redgewise.plot_neighbors import run_plot_neighbors
from redgewise.plot_rave import run_plot_rave
from redgewise.plot_shortest_path import run_plot_shortest_path
from redgewise.plot_vmd import run_plot_vmd


VALUE_CHOICES = ("vdw", "cl", "vdw+cl", "dvdw", "dcl", "dvdw+dcl")
NORMALIZE_CHOICES = (
    "none",
    "per_atom_pair",
    "per_vertex_member_sqrt",
    "per_vertex_member_product",
    "per_coarse_member_sqrt",
    "per_coarse_member_product",
)


def parse_float_or_inf(value: str) -> float:
    normalized = value.strip().lower()
    if normalized in {"inf", "infinity", "none", "off", "false"}:
        return float("inf")
    parsed = float(value)
    if parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be >= 1 or inf")
    return parsed


def add_manual(parser: argparse.ArgumentParser, manual_name: str) -> None:
    parser.add_argument(
        "--man",
        action=ManualAction,
        manual_name=manual_name,
        help="show the full manual for this command and exit",
    )


def add_value_argument(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument(
        "--value",
        choices=VALUE_CHOICES,
        default="vdw+cl",
        help=help_text,
    )


def add_normalize_argument(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument(
        "--normalize",
        choices=NORMALIZE_CHOICES,
        default="none",
        help=help_text,
    )


def add_threshold_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min-abs-value",
        default="none",
        help="minimum abs(analysis value) to keep; number, auto, or none",
    )
    parser.add_argument(
        "--min-abs-percentile",
        type=float,
        default=0.05,
        help="percentile used when --min-abs-value auto; default: 0.05",
    )


def add_exclude_arguments(parser: argparse.ArgumentParser, *, graph_style: bool) -> None:
    if graph_style:
        kind_help = "exclude vertices of this kind before graph/profile construction"
        res_help = "exclude vertices with this residue_name"
        label_help = "exclude vertices with this exact label"
        id_help = "exclude this vertex_id"
    else:
        kind_help = "exclude vertices of this kind from drawing"
        res_help = "exclude vertices with this residue_name from drawing"
        label_help = "exclude vertices with this exact label from drawing"
        id_help = "exclude this vertex_id from drawing"

    parser.add_argument("--exclude-kind", action="append", default=[], metavar="KIND", help=f"{kind_help}; repeatable")
    parser.add_argument("--exclude-resname", action="append", default=[], metavar="RESNAME", help=f"{res_help}; repeatable")
    parser.add_argument("--exclude-label", action="append", default=[], metavar="LABEL", help=f"{label_help}; repeatable")
    parser.add_argument("--exclude-vertex-id", action="append", type=int, default=[], metavar="ID", help=f"{id_help}; repeatable")


def add_remove_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remove-kind", action="append", default=[], metavar="KIND", help="remove vertices of this kind from all neighbor calculations; repeatable")
    parser.add_argument("--remove-resname", action="append", default=[], metavar="RESNAME", help="remove vertices with this residue_name from all neighbor calculations; repeatable")
    parser.add_argument("--remove-label", action="append", default=[], metavar="LABEL", help="remove vertices with this exact label from all neighbor calculations; repeatable")
    parser.add_argument("--remove-vertex-id", action="append", type=int, default=[], metavar="ID", help="remove this vertex_id from all neighbor calculations; repeatable")


def add_profile_display_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--molecule-delimiter-min-size",
        type=parse_float_or_inf,
        default=float("inf"),
        metavar="N",
        help="draw molecule-boundary delimiters for blocks with at least N points; default: inf/off",
    )
    parser.add_argument(
        "--renumber-molecule-residues",
        action="store_true",
        help="label residue ticks as 1..N within each molecule block",
    )


def add_target_argument(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument("--target", default=None, metavar="SELECTOR", help=help_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="redgewise",
        description="Residue Energy edge-wise analysis from GROMACS simulations.",
        epilog="Use `redgewise COMMAND --man` for the full command manual.",
    )
    add_manual(parser, "redgewise")
    parser.add_argument("--version", action="version", version=f"redgewise {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)
    add_build_parser(subparsers)
    add_plot_parser(subparsers)
    add_info_parser(subparsers)
    return parser


def add_build_parser(subparsers: argparse._SubParsersAction) -> None:
    build = subparsers.add_parser(
        "build",
        help="build a redgewise interaction network",
        description="Build a sparse interaction network from GROMACS inputs.",
        epilog="Use `redgewise build --man` for resolution selectors and examples.",
    )
    add_manual(build, "build")
    build.add_argument("-s", "--tpr", type=Path, required=True, help="TPR/topology structure file")
    build.add_argument("-t", "--trajectory", type=Path, required=True, help="trajectory or coordinate source")
    build.add_argument("-p", "--topology", type=Path, required=True, help="GROMACS topology file")
    build.add_argument("-f", "--mdp", type=Path, required=True, help="MDP file with nonbonded cutoffs")
    build.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    build.add_argument("--stride", type=int, default=1, help="analyze every Nth frame; default: 1")
    build.add_argument("--frames-per-part", type=int, default=10, help="processed frames per values parquet part; default: 10")
    build.add_argument("--Worker", "--workers", dest="workers", type=int, default=os.cpu_count(), help="requested worker count; default: CPU count")
    build.add_argument("--gpu", action="store_true", help="request GPU backend if available")
    build.add_argument("--high_res", "--high-res", action="append", default=[], metavar="SELECTOR", help="matched atoms become atom-level vertices; repeatable")
    build.add_argument("--low_res", "--low-res", action="append", default=[], metavar="SELECTOR", help="matched atoms are merged into one low-resolution vertex; repeatable")
    build.add_argument("--bundle", action="append", nargs="+", default=[], metavar="SELECTOR", help="merge listed selectors into one bundle vertex; repeatable")
    build.set_defaults(func=run_build)


def add_plot_parser(subparsers: argparse._SubParsersAction) -> None:
    plot = subparsers.add_parser(
        "plot",
        help="plot or export analyses",
        description="Plot or export analyses from a redgewise build output.",
        epilog="Use `redgewise plot SUBCOMMAND --man` for full plot manuals.",
    )
    add_manual(plot, "plot")
    plot_subparsers = plot.add_subparsers(dest="plot_command", required=True)
    add_plot_vmd_parser(plot_subparsers)
    add_plot_shortest_path_parser(plot_subparsers)
    add_plot_neighbors_parser(plot_subparsers)
    add_plot_rave_parser(plot_subparsers)


def add_plot_vmd_parser(subparsers: argparse._SubParsersAction) -> None:
    vmd = subparsers.add_parser(
        "vmd",
        help="export VMD pseudo-bead and TCL files",
        description="Write a VMD TCL drawer and pseudo-bead PDB from a redgewise output.",
        epilog="Use `redgewise plot vmd --man` for normalization and examples.",
    )
    add_manual(vmd, "plot_vmd")
    vmd.add_argument("-i", "--input", type=Path, required=True, help="redgewise build output directory")
    vmd.add_argument("-s", "--structure", type=Path, required=True, help="structure file used to compute vertex centers")
    vmd.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    add_value_argument(vmd, "edge value used for cylinder radius; default: vdw+cl")
    vmd.add_argument("--normalize", choices=NORMALIZE_CHOICES, default="per_coarse_member_product", help="normalize values before plotting; default: per_coarse_member_product")
    vmd.add_argument("--radius-mode", choices=("linear", "sqrt", "log"), default="log", help="map abs(value) to radius; default: log")
    vmd.add_argument("--radius-scale", type=float, default=0.05, help="cylinder radius scale; default: 0.05")
    vmd.add_argument("--bead-radius", type=float, default=0.25, help="pseudo-bead VDW radius in TCL; default: 0.25")
    vmd.add_argument("--min-abs-value", default="auto", help="minimum normalized abs(value) to draw; number, auto, or none")
    vmd.add_argument("--min-abs-percentile", type=float, default=0.05, help="percentile used when --min-abs-value auto; default: 0.05")
    vmd.add_argument("--max-edges", type=int, default=None, help="draw only the strongest N edges; default: no limit")
    add_exclude_arguments(vmd, graph_style=False)
    vmd.add_argument("--coordinate-unit", choices=("auto", "angstrom", "nm"), default="auto", help="structure-reader coordinate unit; output is Angstrom; default: auto")
    vmd.set_defaults(func=run_plot_vmd)


def add_plot_shortest_path_parser(subparsers: argparse._SubParsersAction) -> None:
    shortest = subparsers.add_parser(
        "shortest_path",
        help="plot mean shortest-path distance profiles",
        description="Plot source-wise mean shortest-path interaction distances.",
        epilog="Use `redgewise plot shortest_path --man` for target selectors and examples.",
    )
    add_manual(shortest, "plot_shortest_path")
    shortest.add_argument("-i", "--input", type=Path, required=True, help="redgewise build output directory")
    shortest.add_argument("-o", "--output", type=Path, required=True, help="output image path or directory")
    add_value_argument(shortest, "edge value used for graph weighting; default: vdw+cl")
    add_normalize_argument(shortest, "normalize values before graph weighting; default: none")
    add_threshold_arguments(shortest)
    add_exclude_arguments(shortest, graph_style=True)
    add_target_argument(shortest, "average each source only to vertices matching this selector")
    add_profile_display_arguments(shortest)
    shortest.set_defaults(func=run_plot_shortest_path)


def add_plot_neighbors_parser(subparsers: argparse._SubParsersAction) -> None:
    neighbors = subparsers.add_parser(
        "neighbors",
        help="plot direct-neighbor edge summaries",
        description="Plot direct-neighbor average edge values per residue.",
        epilog="Use `redgewise plot neighbors --man` for exclude/remove and split modes.",
    )
    add_manual(neighbors, "plot_neighbors")
    neighbors.add_argument("-i", "--input", type=Path, required=True, help="redgewise build output directory")
    neighbors.add_argument("-o", "--output", type=Path, required=True, help="output image path or directory")
    add_value_argument(neighbors, "edge value summarized over direct neighbors; default: vdw+cl")
    add_normalize_argument(neighbors, "normalize values before neighbor summarization; default: none")
    neighbors.add_argument("--neighbor-summary", choices=("mean", "mean_abs", "sum", "sum_abs", "median", "median_abs"), default="mean_abs", help="direct-neighbor summary statistic; default: mean_abs")
    add_threshold_arguments(neighbors)
    add_exclude_arguments(neighbors, graph_style=True)
    add_remove_arguments(neighbors)
    neighbors.add_argument("--split-excluded", action="store_true", help="split contributions toward excluded vs non-excluded endpoints")
    add_target_argument(neighbors, "average each source only over direct neighbor endpoints matching this selector")
    add_profile_display_arguments(neighbors)
    neighbors.set_defaults(func=run_plot_neighbors)


def add_plot_rave_parser(subparsers: argparse._SubParsersAction) -> None:
    rave = subparsers.add_parser(
        "rave",
        help="plot residue-by-frame region-pair interaction heatmaps",
        description="Plot signed direct-neighbor interactions between selected regions over frames.",
        epilog="Use `redgewise plot rave --man` for region semantics and examples.",
    )
    add_manual(rave, "plot_rave")
    rave.add_argument("-i", "--input", type=Path, required=True, help="redgewise build output directory")
    rave.add_argument("-o", "--output", type=Path, required=True, help="output image path or directory")
    add_value_argument(rave, "per-frame edge value plotted between regions; default: vdw+cl")
    add_normalize_argument(rave, "normalize per-frame edge values before plotting; default: none")
    rave.add_argument("--region", action="append", required=True, metavar="SELECTOR", help="disjoint region selector; repeat at least twice")
    rave.add_argument("--region-label", action="append", default=[], metavar="LABEL", help="optional display label for a region; repeat exactly once per --region")
    rave.add_argument("--alpha", type=float, default=0.9, help="overlay alpha scale for directional matrices; default: 0.9")
    rave.add_argument("--pair-layout", choices=("auto", "overlay", "adjacent"), default="auto", help="directional panel layout; auto overlays compatible residue axes and puts incompatible axes adjacent; default: auto")
    rave.add_argument("--darkmode", action="store_true", help="make only subplot interiors black; axes and figure background remain default")
    rave.set_defaults(func=run_plot_rave)


def add_info_parser(subparsers: argparse._SubParsersAction) -> None:
    info = subparsers.add_parser(
        "info",
        help="show program or output information",
        description="Show program information or summarize a redgewise build output.",
        epilog="Use `redgewise info --man` for selector inventory examples.",
    )
    add_manual(info, "info")
    info.add_argument("-i", "--input", type=Path, default=None, help="optional redgewise build output directory")
    info.add_argument("--list", action="store_true", help="print output summary and vertex table")
    info.add_argument("--selector", action="store_true", help="print selector syntax and observed selector values")
    info.add_argument("--long", action="store_true", help="disable truncation for info output")
    info.set_defaults(func=run_info)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
