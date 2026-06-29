from __future__ import annotations
from dataclasses import dataclass
from redgewise.build_mdp import RedgewiseMdpError, read_mdp_nonbonded_information
from redgewise.build_compute import compute_network, RedgewiseComputeError
from redgewise.build_topology import get_interaction_information, RedgewiseBuildError
import argparse
import time

@dataclass(frozen=True)
class BuildOptions:
    workers: int
    gpu: bool
    stride: int
    high_res: tuple[str, ...]
    low_res: tuple[str, ...]
    bundles: tuple[tuple[str, ...], ...]
    frames_per_part: int

def build_options_from_args(args: argparse.Namespace) -> BuildOptions:
    return BuildOptions(
        workers=args.workers,
        gpu=args.gpu,
        stride=args.stride,
        high_res=tuple(args.high_res),
        low_res=tuple(args.low_res),
        bundles=tuple(tuple(bundle) for bundle in args.bundle),
        frames_per_part=args.frames_per_part,
    )

def run_build(args: argparse.Namespace) -> int:
    start=time.perf_counter()
    options = build_options_from_args(args)

    if options.workers < 1:
        print("redgewise build: error: --workers must be >= 1")
        return 2

    if options.gpu:
        print("redgewise build: warning: GPU backend not implemented yet; using CPU.")

    if options.stride < 1:
        print("redgewise build: error: --stride must be >= 1")
        return 2

    if options.frames_per_part < 1:
        print("redgewise build: error: --frames-per-part must be >= 1")
        return 2

    try:
        interaction_information = get_interaction_information(topology=args.topology,tpr=args.tpr)
        mdp_information = read_mdp_nonbonded_information(args.mdp)      

    except (RedgewiseBuildError, RedgewiseMdpError) as exc:
        print(f"redgewise build: error: {exc}")
        return 2

    print("redgewise build")
    print(f"TPR:        {args.tpr}")
    print(f"Trajectory: {args.trajectory}")
    print(f"Topology:   {args.topology}")
    print(f"MDP:        {args.mdp}")
    print(f"Output:     {args.output}")
    print(f"Stride:     {options.stride}")
    print()

    summary = interaction_information.summary()

    print(f"Residues:   {summary.n_residues}")
    print(f"Atoms:      {summary.n_atoms}")
    print(f"VDW pairs:  {summary.n_vdw_type_pairs}")
    print(f"Excluded atom pairs:  {len(interaction_information.excluded_atom_pairs)}")

    print()
    print(f"rlist:          {mdp_information.rlist}")
    print(f"rcoulomb:       {mdp_information.rcoulomb}")
    print(f"rvdw:           {mdp_information.rvdw}")
    print(f"max cutoff:     {mdp_information.max_cutoff}")
    print(f"vdw switch:     {mdp_information.rvdw_switch}")

    print(f"Workers:    {args.workers}")
    print(f"GPU:        {args.gpu}")
    print(f"High-res selectors: {args.high_res}")
    print(f"Low-res selectors:  {args.low_res}")
    print(f"Bundle selectors:   {args.bundle}")

    try:
        compute_summary = compute_network(
            interaction_information=interaction_information,
            mdp_information=mdp_information,
            tpr=args.tpr,
            trajectory=args.trajectory,
            output=args.output,
            options=options,
        )
    except (RedgewiseBuildError, RedgewiseComputeError, ValueError) as exc:
        print(f"redgewise build: error: {exc}")
        return 2

    print(f"Run took {round(time.perf_counter()-start,2)} seconds.")

    return 0