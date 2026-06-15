from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GroupingInformation:
    vertices: list[dict]
    atom_table: dict[str, np.ndarray]
    atom_to_vertex: np.ndarray
    warnings: tuple[str, ...]


def build_grouping_information(
    interaction_information: Any,
    options: Any,
) -> GroupingInformation:
    atom_table = build_atom_table(interaction_information)

    high_res = set(options.high_res)
    low_res = set(options.low_res)
    bundle_map, bundle_labels, warnings = build_bundle_maps(options.bundles)

    warnings.extend(validate_resolution_overwrites(high_res, low_res, bundle_map))

    vertices: list[dict] = []
    vertex_key_to_id: dict[tuple, int] = {}

    n_atoms = len(atom_table["atom_index"])
    atom_to_vertex = np.empty(n_atoms, dtype=np.int32)

    for atom_index in range(n_atoms):
        row = atom_row(atom_table, atom_index)

        vertex_key, vertex_kind, vertex_label = classify_atom_vertex(
            row=row,
            high_res=high_res,
            low_res=low_res,
            bundle_map=bundle_map,
            bundle_labels=bundle_labels,
        )

        atom_to_vertex[atom_index] = get_or_add_vertex_id(
            key=vertex_key,
            row=row,
            kind=vertex_kind,
            label=vertex_label,
            vertices=vertices,
            vertex_key_to_id=vertex_key_to_id,
        )

    return GroupingInformation(
        vertices=vertices,
        atom_table=atom_table,
        atom_to_vertex=atom_to_vertex,
        warnings=tuple(warnings),
    )


def atom_row(atom_table: dict[str, np.ndarray], index: int) -> dict:
    return {key: values[index] for key, values in atom_table.items()}


def classify_atom_vertex(
    row: dict,
    high_res: set[str],
    low_res: set[str],
    bundle_map: dict[str, int],
    bundle_labels: dict[int, str],
) -> tuple[tuple, str, str]:
    resname = row["residue_name"]

    if resname in high_res:
        return (
            ("atom", int(row["atom_nr"])),
            "atom",
            (
                f"{row['residue_name']}:{row['residue_id']}:"
                f"{row['atom_name']}:{row['atom_nr']}"
            ),
        )

    if resname in low_res:
        return (
            ("low_res", resname),
            "low_res",
            resname,
        )

    if resname in bundle_map:
        bundle_id = bundle_map[resname]
        return (
            ("bundle", bundle_id),
            "bundle",
            bundle_labels[bundle_id],
        )

    return (
        ("residue", int(row["residue_id"])),
        "residue",
        f"{row['residue_name']}:{row['residue_id']}",
    )


def get_or_add_vertex_id(
    key: tuple,
    row: dict,
    kind: str,
    label: str,
    vertices: list[dict],
    vertex_key_to_id: dict[tuple, int],
) -> int:
    if key in vertex_key_to_id:
        return vertex_key_to_id[key]

    vertex_id = len(vertices)
    vertex_key_to_id[key] = vertex_id

    vertices.append(
        {
            "vertex_id": vertex_id,
            "label": label,
            "kind": kind,
            "residue_name": row["residue_name"],
            "residue_id": row["residue_id"],
            "molecule_type": row["molecule_type"],
            "molecule_instance": row["molecule_instance"],
            "atom_nr": row["atom_nr"] if kind == "atom" else None,
            "atom_name": row["atom_name"] if kind == "atom" else None,
            "atom_type": row["atom_type"] if kind == "atom" else None,
            "charge": row["charge"] if kind == "atom" else None,
            "members": None,
        }
    )

    return vertex_id


def build_atom_table(interaction_information: Any) -> dict[str, np.ndarray]:
    rows: list[dict] = []

    for residue in interaction_information.residues.values():
        for atom in residue.atoms:
            rows.append(
                {
                    "atom_index": atom.nr - 1,
                    "atom_nr": atom.nr,
                    "residue_id": residue.residue_id,
                    "residue_name": residue.residue_name,
                    "molecule_type": residue.molecule_type,
                    "molecule_instance": residue.molecule_instance,
                    "atom_name": atom.atom_name,
                    "atom_type": atom.atom_type,
                    "charge": atom.charge,
                }
            )

    rows.sort(key=lambda row: row["atom_index"])

    observed = np.array([row["atom_index"] for row in rows], dtype=np.int64)
    expected = np.arange(len(rows), dtype=np.int64)

    if not np.array_equal(observed, expected):
        raise ValueError(
            "atom numbering in InteractionInformation is not continuous from 1..N"
        )

    return {
        "atom_index": observed.astype(np.int32),
        "atom_nr": np.array([row["atom_nr"] for row in rows], dtype=np.int32),
        "residue_id": np.array([row["residue_id"] for row in rows], dtype=np.int32),
        "residue_name": np.array([row["residue_name"] for row in rows], dtype=object),
        "molecule_type": np.array([row["molecule_type"] for row in rows], dtype=object),
        "molecule_instance": np.array(
            [row["molecule_instance"] for row in rows],
            dtype=np.int32,
        ),
        "atom_name": np.array([row["atom_name"] for row in rows], dtype=object),
        "atom_type": np.array([row["atom_type"] for row in rows], dtype=object),
        "charge": np.array([row["charge"] for row in rows], dtype=np.float64),
    }


def build_bundle_maps(
    bundles: tuple[tuple[str, ...], ...],
) -> tuple[dict[str, int], dict[int, str], list[str]]:
    bundle_map: dict[str, int] = {}
    bundle_labels: dict[int, str] = {}
    warnings: list[str] = []

    for bundle_id, bundle in enumerate(bundles):
        names = tuple(bundle)
        label = "+".join(names)
        bundle_labels[bundle_id] = label

        for name in names:
            if name in bundle_map:
                warnings.append(
                    "resolution warning: residue name "
                    f"{name!r} appears in multiple --bundle definitions; "
                    "using the first bundle."
                )
                continue

            bundle_map[name] = bundle_id

    return bundle_map, bundle_labels, warnings


def validate_resolution_overwrites(
    high_res: set[str],
    low_res: set[str],
    bundle_map: dict[str, int],
) -> list[str]:
    warnings: list[str] = []

    for name in sorted(high_res & low_res):
        warnings.append(
            "resolution warning: residue name "
            f"{name!r} appears in both --high_res and --low_res; "
            "--high_res takes precedence."
        )

    for name in sorted(high_res & set(bundle_map)):
        warnings.append(
            "resolution warning: residue name "
            f"{name!r} appears in --high_res and --bundle; "
            "--high_res takes precedence."
        )

    for name in sorted(low_res & set(bundle_map)):
        warnings.append(
            "resolution warning: residue name "
            f"{name!r} appears in --low_res and --bundle; "
            "--low_res takes precedence."
        )

    return warnings