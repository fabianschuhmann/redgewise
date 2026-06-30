from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from redgewise.analysis_values import compute_edge_analysis_summary


RADIUS_MODES = ("linear", "sqrt", "log")


def run_plot_vmd(args: argparse.Namespace) -> int:
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    structure = args.structure.expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    vertices = pq.read_table(input_dir / "vertices.parquet")
    vertex_members = pq.read_table(input_dir / "vertex_members.parquet")

    edge_summary = compute_edge_analysis_summary(
        input_dir=input_dir,
        value_name=args.value,
        normalization=args.normalize,
        exclude_kinds=getattr(args, "exclude_kind", []),
        exclude_resnames=getattr(args, "exclude_resname", []),
        exclude_labels=getattr(args, "exclude_label", []),
        exclude_vertex_ids=getattr(args, "exclude_vertex_id", []),
        min_abs_value=getattr(args, "min_abs_value", "auto"),
        min_abs_percentile=getattr(args, "min_abs_percentile", 0.05),
        max_edges=getattr(args, "max_edges", None),
    )

    centers_raw, structure_extent_raw = compute_vertex_centers_raw(
        structure=structure,
        vertices=vertices,
        vertex_members=vertex_members,
    )

    centers_angstrom = convert_centers_to_angstrom(
        centers=centers_raw,
        structure_extent=structure_extent_raw,
        coordinate_unit=args.coordinate_unit,
    )

    write_pseudo_bead_pdb(
        path=output_dir / "network_beads.pdb",
        vertices=vertices,
        centers_angstrom=centers_angstrom,
    )

    write_vmd_drawer_tcl(
        path=output_dir / "drawer.tcl",
        edge_table=edge_summary.table,
        centers_angstrom=centers_angstrom,
        radius_scale=args.radius_scale,
        radius_mode=args.radius_mode,
    )

    write_vmd_loader_tcl(
        path=output_dir / "load_network.tcl",
        bead_radius=args.bead_radius,
    )

    write_edge_summary_parquet(
        path=output_dir / "edge_summary.parquet",
        edge_table=edge_summary.table,
    )

    print("redgewise plot vmd")
    print(f"Input:                 {input_dir}")
    print(f"Structure:             {structure}")
    print(f"Output:                {output_dir}")
    print(f"Value:                 {edge_summary.value_name}")
    print(f"Normalization:         {edge_summary.normalization}")
    print(f"Frames averaged:       {edge_summary.n_frames}")
    print(f"Coordinate unit:       {args.coordinate_unit}")
    print(f"Radius mode:           {args.radius_mode}")
    print(f"Radius scale:          {args.radius_scale:g}")
    print(f"Min abs value:         {edge_summary.min_abs_value_used:g} ({edge_summary.min_abs_value_mode})")
    print(f"Edges before filters:  {edge_summary.n_edges_before_filter}")
    print(f"After exclusions:      {edge_summary.n_edges_after_exclusion}")
    print(f"After min abs filter:  {edge_summary.n_edges_after_threshold}")
    print(f"Edges drawn:           {edge_summary.n_edges_final}")
    print()
    print(f"Load in VMD with: source {output_dir / 'load_network.tcl'}")

    return 0


def compute_vertex_centers_raw(
    structure: Path,
    vertices: pa.Table,
    vertex_members: pa.Table,
) -> tuple[np.ndarray, np.ndarray]:
    universe = mda.Universe(str(structure))

    n_vertices = vertices.num_rows
    centers = np.full((n_vertices, 3), np.nan, dtype=np.float64)

    vertex_ids = vertex_members.column("vertex_id").to_numpy(zero_copy_only=False)
    atom_indices = vertex_members.column("atom_index").to_numpy(zero_copy_only=False)

    if len(atom_indices) == 0:
        raise ValueError("vertex_members.parquet contains no atoms")

    if np.min(atom_indices) < 0:
        raise ValueError("vertex_members.parquet contains negative atom indices")

    if np.max(atom_indices) >= len(universe.atoms):
        raise ValueError(
            "vertex_members.parquet contains atom indices that exceed the structure "
            f"atom count: max atom_index={int(np.max(atom_indices))}, "
            f"structure atoms={len(universe.atoms)}"
        )

    order = np.argsort(vertex_ids)
    sorted_vertex_ids = vertex_ids[order]
    sorted_atom_indices = atom_indices[order]

    unique_vertex_ids, starts = np.unique(sorted_vertex_ids, return_index=True)
    stops = np.append(starts[1:], len(sorted_vertex_ids))

    for vertex_id, start, stop in zip(unique_vertex_ids, starts, stops):
        members = sorted_atom_indices[start:stop]
        atoms = universe.atoms[members]
        centers[int(vertex_id)] = center_of_mass_or_geometry(atoms)

    if np.any(~np.isfinite(centers)):
        missing = np.where(~np.isfinite(centers).all(axis=1))[0]
        raise ValueError(
            "could not compute centers for some vertices; first missing vertex_id: "
            f"{int(missing[0])}"
        )

    positions = np.asarray(universe.atoms.positions, dtype=np.float64)
    structure_extent = positions.max(axis=0) - positions.min(axis=0)

    return centers, structure_extent


def center_of_mass_or_geometry(atoms: Any) -> np.ndarray:
    try:
        masses = atoms.masses
        if (
            len(masses) == len(atoms)
            and np.all(np.isfinite(masses))
            and np.sum(masses) > 0.0
        ):
            return np.asarray(atoms.center_of_mass(), dtype=np.float64)
    except Exception:
        pass

    return np.asarray(atoms.positions.mean(axis=0), dtype=np.float64)


def convert_centers_to_angstrom(
    centers: np.ndarray,
    structure_extent: np.ndarray,
    coordinate_unit: str,
) -> np.ndarray:
    if coordinate_unit == "angstrom":
        print("redgewise plot vmd: using structure coordinates as Angstrom.")
        return centers

    if coordinate_unit == "nm":
        print("redgewise plot vmd: converting structure coordinates from nm to Angstrom.")
        return centers * 10.0

    if coordinate_unit != "auto":
        raise ValueError("coordinate_unit must be one of: auto, angstrom, nm")

    max_extent = float(np.max(structure_extent))

    if max_extent < 30.0:
        print(
            "redgewise plot vmd: structure coordinate extent looks like nm "
            f"(max extent {max_extent:.3f}); converting to Angstrom. "
            "Override with --coordinate-unit angstrom if this is wrong."
        )
        return centers * 10.0

    print(
        "redgewise plot vmd: structure coordinate extent looks like Angstrom "
        f"(max extent {max_extent:.3f}); using as-is. "
        "Override with --coordinate-unit nm if this is wrong."
    )
    return centers


def write_pseudo_bead_pdb(
    path: Path,
    vertices: pa.Table,
    centers_angstrom: np.ndarray,
) -> None:
    labels = vertices.column("label").to_pylist()
    residue_names = vertices.column("residue_name").to_pylist()
    kinds = get_optional_column(vertices, "kind", default="vertex")

    with path.open("w") as handle:
        handle.write("REMARK redgewise pseudo-bead vertex centers\n")
        handle.write("REMARK coordinates are written in Angstrom\n")
        handle.write("REMARK atom serial and residue id are vertex_id + 1\n")

        for vertex_id, xyz in enumerate(centers_angstrom):
            resname = sanitize_resname(residue_names[vertex_id])
            atom_name = sanitize_atom_name(kinds[vertex_id])
            resid = vertex_id + 1
            serial = vertex_id + 1
            x, y, z = xyz

            handle.write(
                f"HETATM{serial:5d} {atom_name:^4s} {resname:>3s} A"
                f"{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.00:6.2f}{0.00:6.2f}           X\n"
            )

        handle.write("END\n")

    labels_path = path.with_suffix(".labels.tsv")
    with labels_path.open("w") as handle:
        handle.write("vertex_id\tlabel\n")
        for vertex_id, label in enumerate(labels):
            handle.write(f"{vertex_id}\t{label}\n")


def write_vmd_drawer_tcl(
    path: Path,
    edge_table: pa.Table,
    centers_angstrom: np.ndarray,
    radius_scale: float,
    radius_mode: str,
) -> None:
    if radius_mode not in RADIUS_MODES:
        raise ValueError(f"unknown radius mode {radius_mode!r}; expected one of {RADIUS_MODES}")

    vertex1 = edge_table.column("vertex1").to_numpy(zero_copy_only=False)
    vertex2 = edge_table.column("vertex2").to_numpy(zero_copy_only=False)
    values = edge_table.column("value").to_numpy(zero_copy_only=False)

    with path.open("w") as handle:
        handle.write("# redgewise VMD drawer\n")
        handle.write("# coordinates are in Angstrom, matching network_beads.pdb\n")
        handle.write(f"# radius_mode = {radius_mode}\n")
        handle.write(f"# radius_scale = {radius_scale:g}\n")
        handle.write("draw delete all\n")
        handle.write("draw materials on\n")

        for v1, v2, value in zip(vertex1, vertex2, values):
            radius = radius_from_value(
                value=float(value),
                radius_scale=radius_scale,
                radius_mode=radius_mode,
            )

            if radius <= 0.0 or not math.isfinite(radius):
                continue

            if value < 0:
                handle.write("draw color blue\n")
            else:
                handle.write("draw color red\n")

            p1 = centers_angstrom[int(v1)]
            p2 = centers_angstrom[int(v2)]

            handle.write(
                "draw cylinder "
                f"{{{p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f}}} "
                f"{{{p2[0]:.3f} {p2[1]:.3f} {p2[2]:.3f}}} "
                f"radius {radius:.4f} filled yes resolution 12\n"
            )


def radius_from_value(value: float, radius_scale: float, radius_mode: str) -> float:
    magnitude = abs(value)

    if radius_mode == "linear":
        transformed = magnitude
    elif radius_mode == "sqrt":
        transformed = math.sqrt(magnitude)
    elif radius_mode == "log":
        transformed = math.log1p(magnitude)
    else:
        raise ValueError(f"unknown radius mode: {radius_mode}")

    return transformed * radius_scale


def write_vmd_loader_tcl(path: Path, bead_radius: float) -> None:
    with path.open("w") as handle:
        handle.write("# redgewise VMD loader\n")
        handle.write("set here [file dirname [info script]]\n")
        handle.write("mol new [file join $here network_beads.pdb] type pdb\n")
        handle.write("mol delrep 0 top\n")
        handle.write(f"mol representation VDW {bead_radius:.4f} 12\n")
        handle.write("mol color Name\n")
        handle.write("mol selection all\n")
        handle.write("mol material Opaque\n")
        handle.write("mol addrep top\n")
        handle.write("source [file join $here drawer.tcl]\n")


def write_edge_summary_parquet(path: Path, edge_table: pa.Table) -> None:
    pq.write_table(edge_table, path, compression="zstd")


def get_optional_column(table: pa.Table, name: str, default: object) -> list[object]:
    if name not in table.column_names:
        return [default] * table.num_rows

    return table.column(name).to_pylist()


def sanitize_resname(value: object) -> str:
    if value is None:
        return "VTX"

    text = str(value).strip()

    if not text:
        return "VTX"

    text = "".join(char for char in text if char.isalnum())

    if not text:
        return "VTX"

    return text[:3].upper()


def sanitize_atom_name(value: object) -> str:
    if value is None:
        return "V"

    text = str(value).strip().upper()

    if text.startswith("ATOM"):
        return "A"

    if text.startswith("RES"):
        return "R"

    if text.startswith("LOW"):
        return "L"

    if text.startswith("BUN"):
        return "B"

    return "V"
