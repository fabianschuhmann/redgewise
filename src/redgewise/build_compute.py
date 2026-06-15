
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import MDAnalysis as mda
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from redgewise import __version__
from redgewise.build_groups import build_grouping_information
from redgewise.build_information import pair_key


class RedgewiseComputeError(Exception):
    """An error occured during network computation."""

COULOMB_CONSTANT = 138.935458
ATOM_BLOCK_SIZE = 512

VALUE_SCHEMA = pa.schema(
    [
        ("frame", pa.int32()),
        ("edge_key", pa.int64()),
        ("vdw", pa.float32()),
        ("coulomb", pa.float32()),
        ("vdw_dif", pa.float32()),
        ("coulomb_dif", pa.float32()),
        ("n_atom_pairs", pa.int32()),
    ]
)

EDGE_SCHEMA = pa.schema(
    [
        ("edge_key", pa.int64()),
        ("vertex1", pa.int32()),
        ("vertex2", pa.int32()),
    ]
)


@dataclass(frozen=True)
class ComputeSummary:
    n_atoms: int
    n_vertices: int
    n_frames: int
    n_edges: int


@dataclass
class FrameEdgeAccumulator:
    vdw: dict[int, float] = field(default_factory=dict)
    coulomb: dict[int, float] = field(default_factory=dict)
    vdw_dif: dict[int, float] = field(default_factory=dict)
    coulomb_dif: dict[int, float] = field(default_factory=dict)
    n_atom_pairs: dict[int, int] = field(default_factory=dict)

    def add(
        self,
        edge_key: np.ndarray,
        vdw: np.ndarray,
        coulomb: np.ndarray,
        vdw_dif: np.ndarray,
        coulomb_dif: np.ndarray,
        n_atom_pairs: np.ndarray,
    ) -> None:
        if len(edge_key) == 0:
            return

        for index, key_value in enumerate(edge_key):
            key = int(key_value)

            self.vdw[key] = self.vdw.get(key, 0.0) + float(vdw[index])
            self.coulomb[key] = self.coulomb.get(key, 0.0) + float(coulomb[index])
            self.vdw_dif[key] = self.vdw_dif.get(key, 0.0) + float(vdw_dif[index])
            self.coulomb_dif[key] = (
                self.coulomb_dif.get(key, 0.0) + float(coulomb_dif[index])
            )
            self.n_atom_pairs[key] = (
                self.n_atom_pairs.get(key, 0) + int(n_atom_pairs[index])
            )

    def to_table(self, frame_index: int) -> pa.Table:
        if not self.vdw:
            return empty_value_table()

        edge_keys = np.array(sorted(self.vdw), dtype=np.int64)

        return value_table_from_arrays(
            frame=np.full(len(edge_keys), frame_index, dtype=np.int32),
            edge_key=edge_keys,
            vdw=np.array([self.vdw[int(key)] for key in edge_keys], dtype=np.float32),
            coulomb=np.array(
                [self.coulomb[int(key)] for key in edge_keys],
                dtype=np.float32,
            ),
            vdw_dif=np.array(
                [self.vdw_dif[int(key)] for key in edge_keys],
                dtype=np.float32,
            ),
            coulomb_dif=np.array(
                [self.coulomb_dif[int(key)] for key in edge_keys],
                dtype=np.float32,
            ),
            n_atom_pairs=np.array(
                [self.n_atom_pairs[int(key)] for key in edge_keys],
                dtype=np.int32,
            ),
        )


def compute_network(
    interaction_information: Any,
    mdp_information: Any,
    tpr: Path,
    trajectory: Path,
    output: Path,
    options: Any,
) -> ComputeSummary:
    grouping = build_grouping_information(
        interaction_information=interaction_information,
        options=options,
    )

    for warning in grouping.warnings:
        print(f"redgewise build: warning: {warning}")

    if mdp_information.rvdw_switch is not None:
        print(
            "redgewise build: warning: VDW switching distance detected but "
            "switching is not applied in this first implementation."
        )

    output = output.expanduser().resolve()
    values_dir = output / "values"

    output.mkdir(parents=True, exist_ok=True)
    values_dir.mkdir(parents=True, exist_ok=True)

    pq.write_table(
        pa.Table.from_pylist(grouping.vertices),
        output / "vertices.parquet",
        compression="zstd",
    )

    n_atoms = len(grouping.atom_table["atom_index"])
    n_vertices = len(grouping.vertices)

    atom_type_ids, sigma_by_type, epsilon_by_type = build_type_parameter_arrays(
        atom_table=grouping.atom_table,
        interaction_information=interaction_information,
    )

    charges = grouping.atom_table["charge"].astype(np.float64, copy=False)
    atom_to_vertex = grouping.atom_to_vertex

    universe = mda.Universe(str(tpr), str(trajectory))

    if len(universe.atoms) != n_atoms:
        raise ValueError(
            "trajectory atom count does not match InteractionInformation atom count: "
            f"trajectory={len(universe.atoms)}, interaction_information={n_atoms}"
        )

    frames_per_part = getattr(options, "frames_per_part", 1)
    if frames_per_part < 1:
        raise ValueError("frames_per_part must be >= 1")

    value_parts: list[pa.Table] = []
    frames_in_part = 0
    part_index = 0
    total_value_rows = 0
    total_frames = 0

    cutoff_nm = mdp_information.max_cutoff

    for _, ts in enumerate(universe.trajectory[:: options.stride]):
        frame_values = compute_frame_values_streaming_cells(
            positions_a=universe.atoms.positions,
            box_a=ts.dimensions,
            frame_index=int(ts.frame),
            cutoff_nm=cutoff_nm,
            rcoulomb_nm=mdp_information.rcoulomb,
            rvdw_nm=mdp_information.rvdw,
            atom_type_ids=atom_type_ids,
            sigma_by_type=sigma_by_type,
            epsilon_by_type=epsilon_by_type,
            charges=charges,
            atom_to_vertex=atom_to_vertex,
            n_vertices=n_vertices,
        )

        if frame_values.num_rows > 0:
            value_parts.append(frame_values)
            total_value_rows += frame_values.num_rows

        frames_in_part += 1
        total_frames += 1

        if frames_in_part >= frames_per_part:
            part_index = flush_value_parts(
                value_parts=value_parts,
                values_dir=values_dir,
                part_index=part_index,
            )
            value_parts.clear()
            frames_in_part = 0

    part_index = flush_value_parts(
        value_parts=value_parts,
        values_dir=values_dir,
        part_index=part_index,
    )

    n_edges = write_edges_dictionary(
        values_dir=values_dir,
        output=output,
        n_vertices=n_vertices,
    )

    write_metadata(
        output=output,
        mdp_information=mdp_information,
        options=options,
        n_atoms=n_atoms,
        n_vertices=n_vertices,
        n_edges=n_edges,
        n_value_rows=total_value_rows,
    )

    return ComputeSummary(
        n_atoms=n_atoms,
        n_vertices=n_vertices,
        n_frames=total_frames,
        n_edges=n_edges,
    )


def compute_frame_values_streaming_cells(
    positions_a: np.ndarray,
    box_a: np.ndarray,
    frame_index: int,
    cutoff_nm: float,
    rcoulomb_nm: float,
    rvdw_nm: float,
    atom_type_ids: np.ndarray,
    sigma_by_type: np.ndarray,
    epsilon_by_type: np.ndarray,
    charges: np.ndarray,
    atom_to_vertex: np.ndarray,
    n_vertices: int,
) -> pa.Table:
    positions_nm = np.asarray(positions_a, dtype=np.float64) * 0.1
    box_lengths_nm = orthorhombic_box_lengths_nm(box_a)

    cell_information = build_cell_information(
        positions_nm=positions_nm,
        cutoff_nm=cutoff_nm,
        box_lengths_nm=box_lengths_nm,
    )

    cell_pairs = build_neighbor_cell_pairs(
        occupied_cell_ids=cell_information.cell_atom_indices.keys(),
        n_cells=cell_information.n_cells,
        periodic=cell_information.periodic,
    )

    accumulator = FrameEdgeAccumulator()

    iterator = progress_iterator(
        cell_pairs,
        total=len(cell_pairs),
        desc=f"frame {frame_index} cell-pairs",
        unit="cell-pair",
    )

    for cell_id_a, cell_id_b in iterator:
        atoms_a = cell_information.cell_atom_indices.get(cell_id_a)
        atoms_b = cell_information.cell_atom_indices.get(cell_id_b)

        if atoms_a is None or atoms_b is None:
            continue

        process_cell_pair(
            atoms_a=atoms_a,
            atoms_b=atoms_b,
            same_cell=cell_id_a == cell_id_b,
            positions_nm=positions_nm,
            box_lengths_nm=box_lengths_nm,
            cutoff_nm=cutoff_nm,
            rcoulomb_nm=rcoulomb_nm,
            rvdw_nm=rvdw_nm,
            atom_type_ids=atom_type_ids,
            sigma_by_type=sigma_by_type,
            epsilon_by_type=epsilon_by_type,
            charges=charges,
            atom_to_vertex=atom_to_vertex,
            n_vertices=n_vertices,
            accumulator=accumulator,
        )

    return accumulator.to_table(frame_index=frame_index)


@dataclass(frozen=True)
class CellInformation:
    cell_atom_indices: dict[int, np.ndarray]
    n_cells: tuple[int, int, int]
    periodic: bool


def build_cell_information(
    positions_nm: np.ndarray,
    cutoff_nm: float,
    box_lengths_nm: np.ndarray | None,
) -> CellInformation:
    if box_lengths_nm is not None:
        n_cells = tuple(
            max(1, int(np.floor(length / cutoff_nm)))
            for length in box_lengths_nm
        )
        cell_size_nm = box_lengths_nm / np.array(n_cells, dtype=np.float64)

        wrapped_positions = positions_nm % box_lengths_nm
        cell_coordinates = np.floor(wrapped_positions / cell_size_nm).astype(np.int64)

        for axis, n_axis in enumerate(n_cells):
            cell_coordinates[:, axis] = np.clip(
                cell_coordinates[:, axis],
                0,
                n_axis - 1,
            )

        return CellInformation(
            cell_atom_indices=group_atoms_by_cell(cell_coordinates, n_cells),
            n_cells=n_cells,
            periodic=True,
        )

    origin = positions_nm.min(axis=0)
    shifted_positions = positions_nm - origin
    extent = shifted_positions.max(axis=0)

    n_cells = tuple(
        max(1, int(np.floor(length / cutoff_nm)) + 1)
        for length in extent
    )

    cell_coordinates = np.floor(shifted_positions / cutoff_nm).astype(np.int64)

    for axis, n_axis in enumerate(n_cells):
        cell_coordinates[:, axis] = np.clip(
            cell_coordinates[:, axis],
            0,
            n_axis - 1,
        )

    return CellInformation(
        cell_atom_indices=group_atoms_by_cell(cell_coordinates, n_cells),
        n_cells=n_cells,
        periodic=False,
    )


def group_atoms_by_cell(
    cell_coordinates: np.ndarray,
    n_cells: tuple[int, int, int],
) -> dict[int, np.ndarray]:
    linear_cell_ids = encode_cell_coordinates_array(cell_coordinates, n_cells)
    order = np.argsort(linear_cell_ids)

    sorted_cell_ids = linear_cell_ids[order]
    unique_cell_ids, starts = np.unique(sorted_cell_ids, return_index=True)
    stops = np.append(starts[1:], len(sorted_cell_ids))

    return {
        int(cell_id): order[start:stop].astype(np.int32, copy=False)
        for cell_id, start, stop in zip(unique_cell_ids, starts, stops)
    }


def build_neighbor_cell_pairs(
    occupied_cell_ids: Iterable[int],
    n_cells: tuple[int, int, int],
    periodic: bool,
) -> list[tuple[int, int]]:
    occupied = set(int(cell_id) for cell_id in occupied_cell_ids)
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []

    for cell_id in sorted(occupied):
        ix, iy, iz = decode_cell_id(cell_id, n_cells)

        for dx in (-1, 0, 1):
            nx = ix + dx
            if periodic:
                nx %= n_cells[0]
            elif nx < 0 or nx >= n_cells[0]:
                continue

            for dy in (-1, 0, 1):
                ny = iy + dy
                if periodic:
                    ny %= n_cells[1]
                elif ny < 0 or ny >= n_cells[1]:
                    continue

                for dz in (-1, 0, 1):
                    nz = iz + dz
                    if periodic:
                        nz %= n_cells[2]
                    elif nz < 0 or nz >= n_cells[2]:
                        continue

                    neighbor_id = encode_cell_coordinates(nx, ny, nz, n_cells)

                    if neighbor_id not in occupied:
                        continue

                    pair = ordered_pair(cell_id, neighbor_id)

                    if pair in seen:
                        continue

                    seen.add(pair)
                    pairs.append(pair)

    return pairs


def process_cell_pair(
    atoms_a: np.ndarray,
    atoms_b: np.ndarray,
    same_cell: bool,
    positions_nm: np.ndarray,
    box_lengths_nm: np.ndarray | None,
    cutoff_nm: float,
    rcoulomb_nm: float,
    rvdw_nm: float,
    atom_type_ids: np.ndarray,
    sigma_by_type: np.ndarray,
    epsilon_by_type: np.ndarray,
    charges: np.ndarray,
    atom_to_vertex: np.ndarray,
    n_vertices: int,
    accumulator: FrameEdgeAccumulator,
) -> None:
    if same_cell:
        process_same_cell_pair(
            atoms=atoms_a,
            positions_nm=positions_nm,
            box_lengths_nm=box_lengths_nm,
            cutoff_nm=cutoff_nm,
            rcoulomb_nm=rcoulomb_nm,
            rvdw_nm=rvdw_nm,
            atom_type_ids=atom_type_ids,
            sigma_by_type=sigma_by_type,
            epsilon_by_type=epsilon_by_type,
            charges=charges,
            atom_to_vertex=atom_to_vertex,
            n_vertices=n_vertices,
            accumulator=accumulator,
        )
        return

    for block_a in iter_atom_blocks(atoms_a):
        for block_b in iter_atom_blocks(atoms_b):
            process_atom_block_pair(
                atoms_a=block_a,
                atoms_b=block_b,
                same_atom_block=False,
                positions_nm=positions_nm,
                box_lengths_nm=box_lengths_nm,
                cutoff_nm=cutoff_nm,
                rcoulomb_nm=rcoulomb_nm,
                rvdw_nm=rvdw_nm,
                atom_type_ids=atom_type_ids,
                sigma_by_type=sigma_by_type,
                epsilon_by_type=epsilon_by_type,
                charges=charges,
                atom_to_vertex=atom_to_vertex,
                n_vertices=n_vertices,
                accumulator=accumulator,
            )


def process_same_cell_pair(
    atoms: np.ndarray,
    positions_nm: np.ndarray,
    box_lengths_nm: np.ndarray | None,
    cutoff_nm: float,
    rcoulomb_nm: float,
    rvdw_nm: float,
    atom_type_ids: np.ndarray,
    sigma_by_type: np.ndarray,
    epsilon_by_type: np.ndarray,
    charges: np.ndarray,
    atom_to_vertex: np.ndarray,
    n_vertices: int,
    accumulator: FrameEdgeAccumulator,
) -> None:
    blocks = list(iter_atom_blocks(atoms))

    for block_index_a, block_a in enumerate(blocks):
        for block_index_b in range(block_index_a, len(blocks)):
            block_b = blocks[block_index_b]

            process_atom_block_pair(
                atoms_a=block_a,
                atoms_b=block_b,
                same_atom_block=block_index_a == block_index_b,
                positions_nm=positions_nm,
                box_lengths_nm=box_lengths_nm,
                cutoff_nm=cutoff_nm,
                rcoulomb_nm=rcoulomb_nm,
                rvdw_nm=rvdw_nm,
                atom_type_ids=atom_type_ids,
                sigma_by_type=sigma_by_type,
                epsilon_by_type=epsilon_by_type,
                charges=charges,
                atom_to_vertex=atom_to_vertex,
                n_vertices=n_vertices,
                accumulator=accumulator,
            )


def process_atom_block_pair(
    atoms_a: np.ndarray,
    atoms_b: np.ndarray,
    same_atom_block: bool,
    positions_nm: np.ndarray,
    box_lengths_nm: np.ndarray | None,
    cutoff_nm: float,
    rcoulomb_nm: float,
    rvdw_nm: float,
    atom_type_ids: np.ndarray,
    sigma_by_type: np.ndarray,
    epsilon_by_type: np.ndarray,
    charges: np.ndarray,
    atom_to_vertex: np.ndarray,
    n_vertices: int,
    accumulator: FrameEdgeAccumulator,
) -> None:
    if len(atoms_a) == 0 or len(atoms_b) == 0:
        return

    delta = positions_nm[atoms_a][:, None, :] - positions_nm[atoms_b][None, :, :]

    if box_lengths_nm is not None:
        delta -= box_lengths_nm * np.round(delta / box_lengths_nm)

    distances2_nm = np.einsum("ijk,ijk->ij", delta, delta)
    keep = distances2_nm <= cutoff_nm**2

    if same_atom_block:
        keep &= np.triu(np.ones(keep.shape, dtype=bool), k=1)

    if not np.any(keep):
        return

    row_index, column_index = np.nonzero(keep)

    atom_i = atoms_a[row_index]
    atom_j = atoms_b[column_index]

    vertex_a = atom_to_vertex[atom_i]
    vertex_b = atom_to_vertex[atom_j]

    keep_vertices = vertex_a != vertex_b

    if not np.any(keep_vertices):
        return

    atom_i = atom_i[keep_vertices]
    atom_j = atom_j[keep_vertices]
    vertex_a = vertex_a[keep_vertices]
    vertex_b = vertex_b[keep_vertices]

    distances_nm = np.sqrt(distances2_nm[row_index, column_index][keep_vertices])

    add_atom_pair_energies_to_accumulator(
        atom_i=atom_i,
        atom_j=atom_j,
        vertex_a=vertex_a,
        vertex_b=vertex_b,
        distances_nm=distances_nm,
        rcoulomb_nm=rcoulomb_nm,
        rvdw_nm=rvdw_nm,
        atom_type_ids=atom_type_ids,
        sigma_by_type=sigma_by_type,
        epsilon_by_type=epsilon_by_type,
        charges=charges,
        n_vertices=n_vertices,
        accumulator=accumulator,
    )


def add_atom_pair_energies_to_accumulator(
    atom_i: np.ndarray,
    atom_j: np.ndarray,
    vertex_a: np.ndarray,
    vertex_b: np.ndarray,
    distances_nm: np.ndarray,
    rcoulomb_nm: float,
    rvdw_nm: float,
    atom_type_ids: np.ndarray,
    sigma_by_type: np.ndarray,
    epsilon_by_type: np.ndarray,
    charges: np.ndarray,
    n_vertices: int,
    accumulator: FrameEdgeAccumulator,
) -> None:
    type_i = atom_type_ids[atom_i]
    type_j = atom_type_ids[atom_j]

    sigma = sigma_by_type[type_i, type_j]
    epsilon = epsilon_by_type[type_i, type_j]

    vdw = np.zeros(len(distances_nm), dtype=np.float64)
    coulomb = np.zeros(len(distances_nm), dtype=np.float64)
    vdw_dif = np.zeros(len(distances_nm), dtype=np.float64)
    coulomb_dif = np.zeros(len(distances_nm), dtype=np.float64)

    vdw_mask = distances_nm <= rvdw_nm
    coulomb_mask = distances_nm <= rcoulomb_nm

    if np.any(vdw_mask):
        vdw[vdw_mask], vdw_dif[vdw_mask] = lj_energy_and_derivative(
            r_nm=distances_nm[vdw_mask],
            sigma_nm=sigma[vdw_mask],
            epsilon_kj_mol=epsilon[vdw_mask],
        )

    if np.any(coulomb_mask):
        coulomb[coulomb_mask], coulomb_dif[coulomb_mask] = (
            coulomb_energy_and_derivative(
                r_nm=distances_nm[coulomb_mask],
                qi=charges[atom_i[coulomb_mask]],
                qj=charges[atom_j[coulomb_mask]],
            )
        )

    vertex1 = np.minimum(vertex_a, vertex_b)
    vertex2 = np.maximum(vertex_a, vertex_b)

    edge_key = vertex1.astype(np.int64) * n_vertices + vertex2.astype(np.int64)

    unique_edge_keys, inverse = np.unique(edge_key, return_inverse=True)

    accumulator.add(
        edge_key=unique_edge_keys.astype(np.int64),
        vdw=np.bincount(inverse, weights=vdw).astype(np.float64),
        coulomb=np.bincount(inverse, weights=coulomb).astype(np.float64),
        vdw_dif=np.bincount(inverse, weights=vdw_dif).astype(np.float64),
        coulomb_dif=np.bincount(inverse, weights=coulomb_dif).astype(np.float64),
        n_atom_pairs=np.bincount(inverse).astype(np.int32),
    )


def iter_atom_blocks(atom_indices: np.ndarray) -> Iterable[np.ndarray]:
    for start in range(0, len(atom_indices), ATOM_BLOCK_SIZE):
        stop = min(start + ATOM_BLOCK_SIZE, len(atom_indices))
        yield atom_indices[start:stop]


def lj_energy_and_derivative(
    r_nm: np.ndarray,
    sigma_nm: np.ndarray,
    epsilon_kj_mol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sr = sigma_nm / r_nm
    sr6 = sr**6
    sr12 = sr6 * sr6

    energy = 4.0 * epsilon_kj_mol * (sr12 - sr6)
    derivative = 24.0 * epsilon_kj_mol / r_nm * (-2.0 * sr12 + sr6)

    return energy, derivative


def coulomb_energy_and_derivative(
    r_nm: np.ndarray,
    qi: np.ndarray,
    qj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    qq = qi * qj

    energy = COULOMB_CONSTANT * qq / r_nm
    derivative = -COULOMB_CONSTANT * qq / (r_nm**2)

    return energy, derivative


def build_type_parameter_arrays(
    atom_table: dict[str, np.ndarray],
    interaction_information: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bead_types = sorted(set(atom_table["atom_type"]))
    type_to_id = {bead_type: i for i, bead_type in enumerate(bead_types)}

    atom_type_ids = np.array(
        [type_to_id[atom_type] for atom_type in atom_table["atom_type"]],
        dtype=np.int32,
    )

    n_types = len(bead_types)
    sigma_by_type = np.empty((n_types, n_types), dtype=np.float64)
    epsilon_by_type = np.empty((n_types, n_types), dtype=np.float64)

    for type_i in bead_types:
        for type_j in bead_types:
            key = pair_key(type_i, type_j)

            if key not in interaction_information.vdw_by_type_pair:
                raise ValueError(
                    "missing VDW interaction for bead-type pair: "
                    f"{type_i}, {type_j}"
                )

            interaction = interaction_information.vdw_by_type_pair[key]

            i = type_to_id[type_i]
            j = type_to_id[type_j]

            sigma_by_type[i, j] = interaction.sigma
            epsilon_by_type[i, j] = interaction.epsilon

    return atom_type_ids, sigma_by_type, epsilon_by_type


def value_table_from_arrays(
    frame: np.ndarray,
    edge_key: np.ndarray,
    vdw: np.ndarray,
    coulomb: np.ndarray,
    vdw_dif: np.ndarray,
    coulomb_dif: np.ndarray,
    n_atom_pairs: np.ndarray,
) -> pa.Table:
    return pa.table(
        {
            "frame": pa.array(frame, type=pa.int32()),
            "edge_key": pa.array(edge_key, type=pa.int64()),
            "vdw": pa.array(vdw, type=pa.float32()),
            "coulomb": pa.array(coulomb, type=pa.float32()),
            "vdw_dif": pa.array(vdw_dif, type=pa.float32()),
            "coulomb_dif": pa.array(coulomb_dif, type=pa.float32()),
            "n_atom_pairs": pa.array(n_atom_pairs, type=pa.int32()),
        },
        schema=VALUE_SCHEMA,
    )


def empty_value_table() -> pa.Table:
    return VALUE_SCHEMA.empty_table()


def flush_value_parts(
    value_parts: list[pa.Table],
    values_dir: Path,
    part_index: int,
) -> int:
    nonempty_parts = [table for table in value_parts if table.num_rows > 0]

    if not nonempty_parts:
        return part_index

    values = pa.concat_tables(nonempty_parts)

    path = values_dir / f"part-{part_index:05d}.parquet"
    pq.write_table(values, path, compression="zstd")

    return part_index + 1


def write_edges_dictionary(
    values_dir: Path,
    output: Path,
    n_vertices: int,
) -> int:
    edge_keys: set[int] = set()

    for path in sorted(values_dir.glob("*.parquet")):
        table = pq.read_table(path, columns=["edge_key"])
        edge_keys.update(int(value) for value in table.column("edge_key").to_pylist())

    edge_key_array = np.array(sorted(edge_keys), dtype=np.int64)

    vertex1 = (edge_key_array // n_vertices).astype(np.int32)
    vertex2 = (edge_key_array % n_vertices).astype(np.int32)

    table = pa.table(
        {
            "edge_key": pa.array(edge_key_array, type=pa.int64()),
            "vertex1": pa.array(vertex1, type=pa.int32()),
            "vertex2": pa.array(vertex2, type=pa.int32()),
        },
        schema=EDGE_SCHEMA,
    )

    pq.write_table(
        table,
        output / "edges.parquet",
        compression="zstd",
    )

    return len(edge_key_array)


def orthorhombic_box_lengths_nm(box_a: np.ndarray | None) -> np.ndarray | None:
    if box_a is None:
        return None

    box = np.asarray(box_a, dtype=np.float64)

    if len(box) < 3:
        return None

    if np.any(box[:3] <= 0.0):
        return None

    if len(box) >= 6 and np.any(np.abs(box[3:6] - 90.0) > 1e-4):
        raise ValueError(
            "triclinic boxes are not supported by the streaming cell-list backend yet"
        )

    return box[:3] * 0.1


def encode_cell_coordinates(
    ix: int,
    iy: int,
    iz: int,
    n_cells: tuple[int, int, int],
) -> int:
    return int(ix + n_cells[0] * (iy + n_cells[1] * iz))


def encode_cell_coordinates_array(
    cell_coordinates: np.ndarray,
    n_cells: tuple[int, int, int],
) -> np.ndarray:
    return (
        cell_coordinates[:, 0]
        + n_cells[0] * (cell_coordinates[:, 1] + n_cells[1] * cell_coordinates[:, 2])
    ).astype(np.int64)


def decode_cell_id(
    cell_id: int,
    n_cells: tuple[int, int, int],
) -> tuple[int, int, int]:
    ix = cell_id % n_cells[0]
    rest = cell_id // n_cells[0]
    iy = rest % n_cells[1]
    iz = rest // n_cells[1]

    return int(ix), int(iy), int(iz)


def ordered_pair(a: int, b: int) -> tuple[int, int]:
    if a <= b:
        return a, b
    return b, a


def progress_iterator(
    iterable: list[tuple[int, int]],
    total: int,
    desc: str,
    unit: str,
) -> Iterable[tuple[int, int]]:
    if tqdm is None:
        return iterable

    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=False,
        dynamic_ncols=True,
        mininterval=1.0,
    )


def write_metadata(
    output: Path,
    mdp_information: Any,
    options: Any,
    n_atoms: int,
    n_vertices: int,
    n_edges: int,
    n_value_rows: int,
) -> None:
    metadata = {
        "redgewise_version": __version__,
        "schema": "redgewise_sparse_undirected_per_frame_v1",
        "stride": options.stride,
        "length_unit": "nm",
        "energy_unit": "kJ/mol",
        "derivative_unit": "kJ/mol/nm",
        "coordinates_note": (
            "MDAnalysis coordinates are Angstrom; converted to nm for energies."
        ),
        "distance_backend": "streaming_orthorhombic_cell_list",
        "atom_block_size": ATOM_BLOCK_SIZE,
        "n_atoms": n_atoms,
        "n_vertices": n_vertices,
        "n_edges": n_edges,
        "n_value_rows": n_value_rows,
        "rlist_nm": mdp_information.rlist,
        "rcoulomb_nm": mdp_information.rcoulomb,
        "rvdw_nm": mdp_information.rvdw,
        "max_cutoff_nm": mdp_information.max_cutoff,
        "rvdw_switch_nm": mdp_information.rvdw_switch,
        "vdw_switch_applied": False,
        "network_directed": False,
        "edge_key_rule": (
            "edge_key = vertex1 * n_vertices + vertex2 with vertex1 < vertex2"
        ),
        "edge_values_are_sparse": True,
        "missing_edge_values_are_zero": True,
        "self_interactions_reported": False,
        "gpu_requested": options.gpu,
        "workers_requested": options.workers,
        "workers_used": 1,
        "high_res": list(options.high_res),
        "low_res": list(options.low_res),
        "bundles": [list(bundle) for bundle in options.bundles],
        "resolution_precedence": [
            "high_res",
            "low_res",
            "bundle",
            "default_residue",
        ],
    }

    with (output / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)