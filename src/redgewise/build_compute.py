from __future__ import annotations

import heapq
import json
import shutil
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
    """Expected compute-time error with a user-readable message."""


COULOMB_CONSTANT = 138.935458
ATOM_BLOCK_SIZE = 512
MAX_UNIT_INFERENCE_PAIRS = 20_000
MAX_GEOMETRY_INFERENCE_ATOMS = 2_000
UNIT_TOO_CLOSE_NM = 0.08

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


EDGE_KEY_RUN_SCHEMA = pa.schema(
    [
        ("edge_key", pa.int64()),
    ]
)

EDGE_KEY_MERGE_BATCH_SIZE = 1_000_000
EDGE_KEY_MERGE_FAN_IN = 64


@dataclass(frozen=True)
class ComputeSummary:
    n_atoms: int
    n_vertices: int
    n_frames: int
    n_edges: int


@dataclass(frozen=True)
class CoordinateScaleInference:
    scale_to_nm: float
    inferred_input_unit: str
    method: str
    n_pairs_sampled: int
    median_distance_input: float | None
    median_distance_nm: float | None
    p01_distance_nm: float | None
    p05_distance_nm: float | None
    p50_distance_nm: float | None
    p95_distance_nm: float | None
    too_close_fraction_nm: float | None
    cutoff_fraction_nm: float | None
    note: str


@dataclass(frozen=True)
class DistanceScaleSummary:
    scale_to_nm: float
    inferred_input_unit: str
    median_input: float
    p01_nm: float
    p05_nm: float
    p50_nm: float
    p95_nm: float
    too_close_fraction_nm: float
    cutoff_fraction_nm: float
    score: float


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


def write_vertex_members(
    output: Path,
    atom_table: dict[str, np.ndarray],
    atom_to_vertex: np.ndarray,
) -> None:
    table = pa.table(
        {
            "vertex_id": pa.array(atom_to_vertex.astype(np.int32), type=pa.int32()),
            "atom_index": pa.array(
                atom_table["atom_index"].astype(np.int32),
                type=pa.int32(),
            ),
            "atom_nr": pa.array(
                atom_table["atom_nr"].astype(np.int32),
                type=pa.int32(),
            ),
        }
    )

    pq.write_table(
        table,
        output / "vertex_members.parquet",
        compression="zstd",
    )



def print_coordinate_inference(inference: CoordinateScaleInference) -> None:
    if inference.median_distance_input is None:
        median = "unknown"
    else:
        median = (
            f"{inference.median_distance_input:.6g} input units = "
            f"{inference.median_distance_nm:.6g} nm"
        )

    print(
        "redgewise build: coordinate scale inferred "
        f"from {inference.method}: "
        f"input_unit={inference.inferred_input_unit}, "
        f"scale_to_nm={inference.scale_to_nm:g}, "
        f"median_local_distance={median}"
    )

    if inference.method.startswith("geometry"):
        print(
            "redgewise build: warning: no excluded/bonded local pairs were available "
            "for coordinate scale inference; used nearest-neighbor geometry fallback."
        )


def infer_coordinate_scale_to_nm(
    raw_positions: np.ndarray,
    raw_box_lengths: np.ndarray | None,
    interaction_information: Any,
    n_atoms: int,
    cutoff_nm: float,
) -> CoordinateScaleInference:
    local_pairs, local_pair_source = local_topology_pairs_for_unit_inference(
        interaction_information=interaction_information,
        n_atoms=n_atoms,
    )

    if len(local_pairs) > 0:
        distances_raw = pair_distances_raw(
            positions=raw_positions,
            pairs=sample_pairs(local_pairs, MAX_UNIT_INFERENCE_PAIRS),
            raw_box_lengths=raw_box_lengths,
        )
        distances_raw = distances_raw[np.isfinite(distances_raw) & (distances_raw > 0.0)]

        if len(distances_raw) == 0:
            raise RedgewiseComputeError(
                "could not infer coordinate scale: local topology distances are empty"
            )

        return choose_coordinate_scale(
            distances_raw=distances_raw,
            cutoff_nm=cutoff_nm,
            method=local_pair_source,
            note=(
                "scale inferred from topology-local excluded pairs; these should "
                "include bonded/nrexcl-derived nonbonded exclusions"
            ),
        )

    distances_raw = nearest_neighbor_distances_raw(
        positions=raw_positions,
        raw_box_lengths=raw_box_lengths,
        max_atoms=MAX_GEOMETRY_INFERENCE_ATOMS,
    )
    distances_raw = distances_raw[np.isfinite(distances_raw) & (distances_raw > 0.0)]

    if len(distances_raw) == 0:
        raise RedgewiseComputeError(
            "could not infer coordinate scale: no excluded pairs and no usable "
            "nearest-neighbor distances"
        )

    return choose_coordinate_scale(
        distances_raw=distances_raw,
        cutoff_nm=cutoff_nm,
        method="geometry_nearest_neighbors",
        note=(
            "scale inferred from nearest-neighbor geometry because no excluded "
            "or bonded local pair list was available"
        ),
    )


def local_topology_pairs_for_unit_inference(
    interaction_information: Any,
    n_atoms: int,
) -> tuple[np.ndarray, str]:
    # In the current build pipeline, excluded_atom_pairs is the authoritative
    # local-topology pair set used by compute. It should include bonded,
    # constraint, explicit exclusion, and nrexcl-derived pairs that must not be
    # evaluated as nonbonded interactions.
    excluded_pairs = getattr(interaction_information, "excluded_atom_pairs", set())
    pairs = sanitize_pairs(excluded_pairs, n_atoms=n_atoms)

    if len(pairs) > 0:
        return pairs, "excluded_atom_pairs"

    # Future-proofing: if build_topology later exposes direct bonds separately,
    # those are an even cleaner coordinate-scale reference. This branch is only
    # reached if excluded_atom_pairs is unavailable/empty.
    bonded_pairs = getattr(interaction_information, "bonded_atom_pairs", set())
    pairs = sanitize_pairs(bonded_pairs, n_atoms=n_atoms)

    if len(pairs) > 0:
        return pairs, "bonded_atom_pairs"

    return np.empty((0, 2), dtype=np.int64), "none"


def sanitize_pairs(
    pairs: Iterable[tuple[int, int]],
    n_atoms: int,
) -> np.ndarray:
    cleaned: set[tuple[int, int]] = set()

    for atom_i, atom_j in pairs:
        i = int(atom_i)
        j = int(atom_j)

        if i == j:
            continue

        if i < 0 or j < 0 or i >= n_atoms or j >= n_atoms:
            continue

        if i > j:
            i, j = j, i

        cleaned.add((i, j))

    if not cleaned:
        return np.empty((0, 2), dtype=np.int64)

    return np.array(sorted(cleaned), dtype=np.int64)


def sample_pairs(pairs: np.ndarray, max_pairs: int) -> np.ndarray:
    if len(pairs) <= max_pairs:
        return pairs

    # Deterministic spread through sorted pairs. Avoid random output changes in
    # metadata and tests.
    indices = np.linspace(0, len(pairs) - 1, num=max_pairs, dtype=np.int64)
    return pairs[indices]


def pair_distances_raw(
    positions: np.ndarray,
    pairs: np.ndarray,
    raw_box_lengths: np.ndarray | None,
) -> np.ndarray:
    delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]

    if raw_box_lengths is not None:
        delta -= raw_box_lengths * np.round(delta / raw_box_lengths)

    return np.linalg.norm(delta, axis=1)


def nearest_neighbor_distances_raw(
    positions: np.ndarray,
    raw_box_lengths: np.ndarray | None,
    max_atoms: int,
) -> np.ndarray:
    n_atoms = len(positions)

    if n_atoms < 2:
        return np.empty(0, dtype=np.float64)

    if n_atoms <= max_atoms:
        sample_indices = np.arange(n_atoms, dtype=np.int64)
    else:
        sample_indices = np.linspace(0, n_atoms - 1, num=max_atoms, dtype=np.int64)

    sample_positions = positions[sample_indices]
    nearest = np.full(len(sample_indices), np.inf, dtype=np.float64)

    block_size = 256
    for start in range(0, len(sample_indices), block_size):
        stop = min(start + block_size, len(sample_indices))
        block = sample_positions[start:stop]

        delta = block[:, None, :] - sample_positions[None, :, :]
        if raw_box_lengths is not None:
            delta -= raw_box_lengths * np.round(delta / raw_box_lengths)

        distances = np.linalg.norm(delta, axis=2)

        row_indices = np.arange(start, stop) - start
        distances[row_indices, np.arange(start, stop)] = np.inf
        nearest[start:stop] = np.min(distances, axis=1)

    return nearest[np.isfinite(nearest)]


def choose_coordinate_scale(
    distances_raw: np.ndarray,
    cutoff_nm: float,
    method: str,
    note: str,
) -> CoordinateScaleInference:
    nm_summary = summarize_coordinate_scale_candidate(
        distances_raw=distances_raw,
        scale_to_nm=1.0,
        inferred_input_unit="nm",
        cutoff_nm=cutoff_nm,
    )
    angstrom_summary = summarize_coordinate_scale_candidate(
        distances_raw=distances_raw,
        scale_to_nm=0.1,
        inferred_input_unit="angstrom",
        cutoff_nm=cutoff_nm,
    )

    if nm_summary.score < angstrom_summary.score:
        chosen = nm_summary
        rejected = angstrom_summary
    else:
        chosen = angstrom_summary
        rejected = nm_summary

    if not np.isfinite(chosen.score):
        raise RedgewiseComputeError("could not infer coordinate scale: invalid score")

    # If both hypotheses are nearly equivalent, do not guess silently. In normal
    # x10 unit mistakes the scores differ by orders of magnitude because one
    # candidate creates impossible <0.08 nm local distances or >1.5 nm local bonds.
    if abs(chosen.score - rejected.score) < 1.0:
        raise RedgewiseComputeError(
            "could not infer coordinate scale unambiguously: "
            f"nm_score={nm_summary.score:.6g}, "
            f"angstrom_score={angstrom_summary.score:.6g}, "
            f"median_input_distance={float(np.median(distances_raw)):.6g}"
        )

    return CoordinateScaleInference(
        scale_to_nm=chosen.scale_to_nm,
        inferred_input_unit=chosen.inferred_input_unit,
        method=method,
        n_pairs_sampled=int(len(distances_raw)),
        median_distance_input=chosen.median_input,
        median_distance_nm=chosen.p50_nm,
        p01_distance_nm=chosen.p01_nm,
        p05_distance_nm=chosen.p05_nm,
        p50_distance_nm=chosen.p50_nm,
        p95_distance_nm=chosen.p95_nm,
        too_close_fraction_nm=chosen.too_close_fraction_nm,
        cutoff_fraction_nm=chosen.cutoff_fraction_nm,
        note=(
            f"{note}; chosen {chosen.inferred_input_unit} "
            f"(score={chosen.score:.6g}) over {rejected.inferred_input_unit} "
            f"(score={rejected.score:.6g})"
        ),
    )


def summarize_coordinate_scale_candidate(
    distances_raw: np.ndarray,
    scale_to_nm: float,
    inferred_input_unit: str,
    cutoff_nm: float,
) -> DistanceScaleSummary:
    distances_nm = distances_raw * scale_to_nm

    p01 = float(np.percentile(distances_nm, 1.0))
    p05 = float(np.percentile(distances_nm, 5.0))
    p50 = float(np.percentile(distances_nm, 50.0))
    p95 = float(np.percentile(distances_nm, 95.0))
    too_close_fraction = float(np.mean(distances_nm < UNIT_TOO_CLOSE_NM))
    cutoff_fraction = float(np.mean(distances_nm <= cutoff_nm))

    score = coordinate_scale_score(
        p01_nm=p01,
        p05_nm=p05,
        p50_nm=p50,
        p95_nm=p95,
        too_close_fraction_nm=too_close_fraction,
        cutoff_fraction_nm=cutoff_fraction,
        cutoff_nm=cutoff_nm,
    )

    return DistanceScaleSummary(
        scale_to_nm=scale_to_nm,
        inferred_input_unit=inferred_input_unit,
        median_input=float(np.median(distances_raw)),
        p01_nm=p01,
        p05_nm=p05,
        p50_nm=p50,
        p95_nm=p95,
        too_close_fraction_nm=too_close_fraction,
        cutoff_fraction_nm=cutoff_fraction,
        score=score,
    )


def coordinate_scale_score(
    p01_nm: float,
    p05_nm: float,
    p50_nm: float,
    p95_nm: float,
    too_close_fraction_nm: float,
    cutoff_fraction_nm: float,
    cutoff_nm: float,
) -> float:
    score = 0.0

    # Hard penalty for impossible local distances. This catches the TPR-as-nm
    # path being erroneously scaled by 0.1, where 0.4 nm contacts become 0.04 nm.
    score += 10_000.0 * too_close_fraction_nm

    if p50_nm < 0.12:
        score += 2_000.0 * (0.12 - p50_nm) / 0.12

    if p05_nm < 0.05:
        score += 1_000.0 * (0.05 - p05_nm) / 0.05

    # Local topology pairs from nrexcl can include more than direct bonds, but a
    # median above ~1.5 nm is not plausible for excluded/bonded-local pairs in
    # this context and indicates Angstrom-like numeric coordinates treated as nm.
    if p50_nm > 1.5:
        score += 2_000.0 * (p50_nm - 1.5) / 1.5

    if p95_nm > max(2.5, 2.0 * cutoff_nm):
        score += 500.0 * (p95_nm - max(2.5, 2.0 * cutoff_nm))

    # Soft preference for local distances around Martini/all-atom topology scale.
    # The score remains broad; the hard x10 penalties above drive the decision.
    reference_nm = 0.35
    score += abs(np.log(max(p50_nm, 1e-12) / reference_nm))

    # Geometry fallback only: if almost no nearest neighbors are within cutoff,
    # the coordinates are probably too large by x10. For excluded pairs this is
    # normally 1.0 for both plausible and correct scales, so it is a weak term.
    if cutoff_fraction_nm < 0.005:
        score += 100.0

    return float(score)


#DIAGNOSTIC
import gc
import os
import resource


def diagnostics_enabled() -> bool:
    value = os.environ.get("REDGEWISE_DIAG_MEMORY", "")
    return value.lower() in {"1", "true", "yes", "on"}


def current_rss_mb() -> float:
    """Current resident set size in MB on Linux.

    Falls back to peak RSS if /proc is unavailable.
    """
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    # Example: VmRSS:  123456 kB
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass

    return peak_rss_mb()


def peak_rss_mb() -> float:
    """Peak resident set size in MB.

    Linux reports ru_maxrss in KiB.
    """
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def arrow_table_nbytes(table) -> int:
    """Approximate Arrow table buffer size in bytes."""
    try:
        return int(table.nbytes)
    except Exception:
        return 0


def diagnostic_log(message: str) -> None:
    if diagnostics_enabled():
        print(message, flush=True)


def diagnostic_memory_log(prefix: str) -> None:
    if not diagnostics_enabled():
        return
    diagnostic_log(
        f"[redgewise memory] {prefix} "
        f"rss={current_rss_mb():.1f}MB "
        f"peak={peak_rss_mb():.1f}MB"
    )
### END DIAGNOSTIC

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
    write_vertex_members(
        output=output,
        atom_table=grouping.atom_table,
        atom_to_vertex=grouping.atom_to_vertex,
    )

    n_atoms = len(grouping.atom_table["atom_index"])
    n_vertices = len(grouping.vertices)

    atom_type_ids, sigma_by_type, epsilon_by_type = build_type_parameter_arrays(
        atom_table=grouping.atom_table,
        interaction_information=interaction_information,
    )

    excluded_neighbors = build_excluded_neighbors(
        excluded_atom_pairs=getattr(interaction_information, "excluded_atom_pairs", set()),
        n_atoms=n_atoms,
    )

    atom_molecule_instance = grouping.atom_table["molecule_instance"]

    charges = grouping.atom_table["charge"].astype(np.float64, copy=False)
    atom_to_vertex = grouping.atom_to_vertex

    try:
        universe = mda.Universe(str(tpr), str(trajectory))
    except Exception as exc:
        raise RedgewiseComputeError(
            f"could not load trajectory with MDAnalysis: {trajectory}"
        ) from exc

    if len(universe.atoms) != n_atoms:
        raise RedgewiseComputeError(
            "trajectory atom count does not match InteractionInformation atom count: "
            f"trajectory={len(universe.atoms)}, interaction_information={n_atoms}"
        )

    cutoff_nm = mdp_information.max_cutoff

    try:
        first_ts = universe.trajectory[0]
    except Exception as exc:
        raise RedgewiseComputeError("could not read first trajectory frame") from exc

    coordinate_inference = infer_coordinate_scale_to_nm(
        raw_positions=np.asarray(universe.atoms.positions, dtype=np.float64),
        raw_box_lengths=orthorhombic_box_lengths_raw(first_ts.dimensions),
        interaction_information=interaction_information,
        n_atoms=n_atoms,
        cutoff_nm=cutoff_nm,
    )

    print_coordinate_inference(coordinate_inference)

    frames_per_part = getattr(options, "frames_per_part", 1)
    if frames_per_part < 1:
        raise RedgewiseComputeError("--frames-per-part must be >= 1")

    value_parts: list[pa.Table] = []
    frames_in_part = 0
    part_index = 0
    total_value_rows = 0
    total_frames = 0

    for processed_frame_index, ts in enumerate(universe.trajectory[:: options.stride]):
        frame_values = compute_frame_values_streaming_cells(
            raw_positions=universe.atoms.positions,
            raw_box=ts.dimensions,
            coordinate_scale_to_nm=coordinate_inference.scale_to_nm,
            frame_index=frame_index_from_ts(ts, fallback=processed_frame_index),
            cutoff_nm=cutoff_nm,
            rcoulomb_nm=mdp_information.rcoulomb,
            rvdw_nm=mdp_information.rvdw,
            atom_type_ids=atom_type_ids,
            sigma_by_type=sigma_by_type,
            epsilon_by_type=epsilon_by_type,
            charges=charges,
            atom_to_vertex=atom_to_vertex,
            n_vertices=n_vertices,
            atom_molecule_instance=atom_molecule_instance,
            excluded_neighbors=excluded_neighbors,
        )
        ###DIAGNOSTIC
        if diagnostics_enabled():
            frame_edge_keys = frame_values.column("edge_key").to_numpy(zero_copy_only=False)
            n_unique_edges_this_frame = int(len(np.unique(frame_edge_keys))) if len(frame_edge_keys) else 0

            diagnostic_log(
                "[redgewise memory] "
                f"frame={frame_index_from_ts(ts, fallback=processed_frame_index)} "
                f"processed_frame={processed_frame_index} "
                f"rows={frame_values.num_rows} "
                f"unique_edges_this_frame={n_unique_edges_this_frame} "
                f"table_bytes={arrow_table_nbytes(frame_values) / (1024.0 * 1024.0):.1f}MB "
                f"rss={current_rss_mb():.1f}MB "
                f"peak={peak_rss_mb():.1f}MB"
            )
        ### END DIAGNOSTIC

        if frame_values.num_rows > 0:
            value_parts.append(frame_values)
            total_value_rows += frame_values.num_rows

        frames_in_part += 1
        total_frames += 1

        if frames_in_part >= frames_per_part:
            ###DIAGNOSTIC
            diagnostic_memory_log(
                f"before flush part={part_index} buffered_tables={len(value_parts)}"
            )
            ### END DIAGONSTIC
            part_index = flush_value_parts(
                value_parts=value_parts,
                values_dir=values_dir,
                part_index=part_index,
            )
            value_parts.clear()
            frames_in_part = 0
            ###DIAGNOSTIC
            gc.collect()

            diagnostic_memory_log(
                f"after flush part={part_index - 1}"
            )
            ###END DIAGNOSTIC

    flush_value_parts(
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
        n_excluded_atom_pairs=sum(len(neighbors) for neighbors in excluded_neighbors) // 2,
        coordinate_inference=coordinate_inference,
    )

    return ComputeSummary(
        n_atoms=n_atoms,
        n_vertices=n_vertices,
        n_frames=total_frames,
        n_edges=n_edges,
    )


def compute_frame_values_streaming_cells(
    raw_positions: np.ndarray,
    raw_box: np.ndarray | None,
    coordinate_scale_to_nm: float,
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
    atom_molecule_instance: np.ndarray,
    excluded_neighbors: list[set[int]],
) -> pa.Table:
    positions_nm = np.asarray(raw_positions, dtype=np.float64) * coordinate_scale_to_nm
    raw_box_lengths = orthorhombic_box_lengths_raw(raw_box)
    box_lengths_nm = (
        None
        if raw_box_lengths is None
        else raw_box_lengths * coordinate_scale_to_nm
    )

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
            atom_molecule_instance=atom_molecule_instance,
            excluded_neighbors=excluded_neighbors,
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
    atom_molecule_instance: np.ndarray,
    excluded_neighbors: list[set[int]],
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
            atom_molecule_instance=atom_molecule_instance,
            excluded_neighbors=excluded_neighbors,
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
                atom_molecule_instance=atom_molecule_instance,
                excluded_neighbors=excluded_neighbors,
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
    atom_molecule_instance: np.ndarray,
    excluded_neighbors: list[set[int]],
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
                atom_molecule_instance=atom_molecule_instance,
                excluded_neighbors=excluded_neighbors,
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
    atom_molecule_instance: np.ndarray,
    excluded_neighbors: list[set[int]],
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

    keep_nonexcluded = nonexcluded_same_molecule_mask(
        atom_i=atom_i,
        atom_j=atom_j,
        atom_molecule_instance=atom_molecule_instance,
        excluded_neighbors=excluded_neighbors,
    )

    if not np.any(keep_nonexcluded):
        return

    atom_i = atom_i[keep_nonexcluded]
    atom_j = atom_j[keep_nonexcluded]
    vertex_a = vertex_a[keep_nonexcluded]
    vertex_b = vertex_b[keep_nonexcluded]
    distances_nm = distances_nm[keep_nonexcluded]

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

def build_excluded_neighbors(
    excluded_atom_pairs: set[tuple[int, int]],
    n_atoms: int,
) -> list[set[int]]:
    neighbors: list[set[int]] = [set() for _ in range(n_atoms)]

    for atom_i, atom_j in excluded_atom_pairs:
        atom_i = int(atom_i)
        atom_j = int(atom_j)

        if atom_i < 0 or atom_j < 0:
            continue

        if atom_i >= n_atoms or atom_j >= n_atoms:
            continue

        if atom_i == atom_j:
            continue

        neighbors[atom_i].add(atom_j)
        neighbors[atom_j].add(atom_i)

    return neighbors


def nonexcluded_same_molecule_mask(
    atom_i: np.ndarray,
    atom_j: np.ndarray,
    atom_molecule_instance: np.ndarray,
    excluded_neighbors: list[set[int]],
) -> np.ndarray:
    keep = np.ones(len(atom_i), dtype=bool)

    same_molecule = (
        atom_molecule_instance[atom_i]
        == atom_molecule_instance[atom_j]
    )

    same_indices = np.nonzero(same_molecule)[0]

    for index in same_indices:
        atom_a = int(atom_i[index])
        atom_b = int(atom_j[index])

        if atom_b in excluded_neighbors[atom_a]:
            keep[index] = False

    return keep

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
                raise RedgewiseComputeError(
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
    """Write edges.parquet without building a global Python set of edge keys.

    The values files already contain the edge_key column. This function builds
    the global unique edge dictionary using an external sorted-unique merge:

    1. Each values part is reduced to a sorted unique edge-key run.
    2. Runs are merged in bounded-fan-in rounds.
    3. The final sorted unique edge-key run is streamed into edges.parquet.

    This keeps memory bounded by one values part plus at most
    EDGE_KEY_MERGE_FAN_IN run batches, instead of storing all unique edge keys
    as Python int objects in a set.
    """

    temp_dir = output / "_redgewise_edge_key_runs"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_memory_log("before external edge dictionary construction")

    try:
        value_parts = sorted(values_dir.glob("*.parquet"))
        if not value_parts:
            pq.write_table(
                EDGE_SCHEMA.empty_table(),
                output / "edges.parquet",
                compression="zstd",
            )
            diagnostic_memory_log("no value parts; wrote empty edges.parquet")
            return 0

        runs: list[Path] = []

        for part_index, values_path in enumerate(value_parts):
            run_path = temp_dir / f"edge-keys-run-{part_index:05d}.parquet"

            diagnostic_memory_log(
                f"before unique edge-key run part={part_index} file={values_path.name}"
            )

            n_rows, n_unique = write_unique_edge_key_run_from_values_part(
                values_path=values_path,
                run_path=run_path,
            )
            runs.append(run_path)

            diagnostic_log(
                "[redgewise memory] "
                f"edge_key_run_part={part_index} "
                f"file={values_path.name} "
                f"rows={n_rows} "
                f"unique_in_part={n_unique} "
                f"run_file={run_path.name} "
                f"rss={current_rss_mb():.1f}MB "
                f"peak={peak_rss_mb():.1f}MB"
            )

            gc.collect()

            diagnostic_memory_log(
                f"after unique edge-key run part={part_index} unique_in_part={n_unique}"
            )

        round_index = 0
        while len(runs) > 1:
            diagnostic_memory_log(
                f"before edge-key merge round={round_index} n_runs={len(runs)}"
            )

            next_runs: list[Path] = []
            for group_index, start_index in enumerate(range(0, len(runs), EDGE_KEY_MERGE_FAN_IN)):
                group = runs[start_index : start_index + EDGE_KEY_MERGE_FAN_IN]
                merged_path = temp_dir / (
                    f"edge-keys-merge-r{round_index:03d}-g{group_index:05d}.parquet"
                )

                n_unique = merge_sorted_edge_key_runs(
                    input_paths=group,
                    output_path=merged_path,
                    batch_size=EDGE_KEY_MERGE_BATCH_SIZE,
                )

                diagnostic_log(
                    "[redgewise memory] "
                    f"edge_key_merge_round={round_index} "
                    f"group={group_index} "
                    f"input_runs={len(group)} "
                    f"unique_out={n_unique} "
                    f"rss={current_rss_mb():.1f}MB "
                    f"peak={peak_rss_mb():.1f}MB"
                )

                next_runs.append(merged_path)

                for path in group:
                    try:
                        path.unlink()
                    except OSError:
                        pass

                gc.collect()

            runs = next_runs
            diagnostic_memory_log(
                f"after edge-key merge round={round_index} n_runs={len(runs)}"
            )
            round_index += 1

        final_run = runs[0]
        diagnostic_memory_log(f"before writing edges.parquet from {final_run.name}")

        n_edges = write_edges_from_unique_edge_key_run(
            run_path=final_run,
            output=output,
            n_vertices=n_vertices,
            batch_size=EDGE_KEY_MERGE_BATCH_SIZE,
        )

        diagnostic_memory_log(f"after writing edges.parquet n_edges={n_edges}")

        return int(n_edges)

    finally:
        if diagnostics_enabled() and os.environ.get("REDGEWISE_KEEP_EDGE_KEY_RUNS", "").lower() in {"1", "true", "yes", "on"}:
            diagnostic_log(f"[redgewise memory] keeping temporary edge-key runs: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


def write_unique_edge_key_run_from_values_part(
    values_path: Path,
    run_path: Path,
) -> tuple[int, int]:
    """Read one values part and write its sorted unique edge keys."""

    table = pq.read_table(values_path, columns=["edge_key"])
    keys = (
        table.column("edge_key")
        .to_numpy(zero_copy_only=False)
        .astype(np.int64, copy=False)
    )

    n_rows = int(len(keys))
    unique_keys = np.unique(keys) if n_rows else np.empty(0, dtype=np.int64)

    run_table = pa.table(
        {
            "edge_key": pa.array(unique_keys, type=pa.int64()),
        },
        schema=EDGE_KEY_RUN_SCHEMA,
    )
    pq.write_table(run_table, run_path, compression="zstd")

    del keys
    del unique_keys
    del table
    del run_table

    return n_rows, int(pq.read_metadata(run_path).num_rows)


class SortedEdgeKeyRunReader:
    """Streaming reader for one sorted edge-key run."""

    def __init__(self, path: Path, batch_size: int):
        self.path = path
        self._batches = pq.ParquetFile(path).iter_batches(
            batch_size=batch_size,
            columns=["edge_key"],
        )
        self._keys = np.empty(0, dtype=np.int64)
        self._position = 0
        self._exhausted = False

    def next_key(self) -> int | None:
        while self._position >= len(self._keys):
            if self._exhausted:
                return None

            try:
                batch = next(self._batches)
            except StopIteration:
                self._exhausted = True
                self._keys = np.empty(0, dtype=np.int64)
                self._position = 0
                return None

            self._keys = (
                batch.column(0)
                .to_numpy(zero_copy_only=False)
                .astype(np.int64, copy=False)
            )
            self._position = 0

            if len(self._keys) == 0:
                continue

        key = int(self._keys[self._position])
        self._position += 1
        return key


def merge_sorted_edge_key_runs(
    input_paths: list[Path],
    output_path: Path,
    batch_size: int,
) -> int:
    """Merge sorted unique edge-key runs into one sorted unique run."""

    if not input_paths:
        pq.write_table(
            EDGE_KEY_RUN_SCHEMA.empty_table(),
            output_path,
            compression="zstd",
        )
        return 0

    readers = [
        SortedEdgeKeyRunReader(path=path, batch_size=batch_size)
        for path in input_paths
    ]

    heap: list[tuple[int, int]] = []
    for reader_index, reader in enumerate(readers):
        key = reader.next_key()
        if key is not None:
            heapq.heappush(heap, (key, reader_index))

    writer = pq.ParquetWriter(
        output_path,
        EDGE_KEY_RUN_SCHEMA,
        compression="zstd",
    )

    output_buffer: list[int] = []
    previous_key: int | None = None
    n_unique = 0
    wrote_any = False

    def flush_output_buffer() -> None:
        nonlocal output_buffer, wrote_any
        if not output_buffer:
            return

        keys = np.asarray(output_buffer, dtype=np.int64)
        table = pa.table(
            {
                "edge_key": pa.array(keys, type=pa.int64()),
            },
            schema=EDGE_KEY_RUN_SCHEMA,
        )
        writer.write_table(table)
        wrote_any = True
        output_buffer = []

    try:
        while heap:
            key, reader_index = heapq.heappop(heap)

            next_key = readers[reader_index].next_key()
            if next_key is not None:
                heapq.heappush(heap, (next_key, reader_index))

            if previous_key is not None and key == previous_key:
                continue

            previous_key = key
            output_buffer.append(key)
            n_unique += 1

            if len(output_buffer) >= batch_size:
                flush_output_buffer()

        flush_output_buffer()

        if not wrote_any:
            writer.write_table(EDGE_KEY_RUN_SCHEMA.empty_table())

    finally:
        writer.close()

    return int(n_unique)


def write_edges_from_unique_edge_key_run(
    run_path: Path,
    output: Path,
    n_vertices: int,
    batch_size: int,
) -> int:
    """Stream one sorted unique edge-key run into edges.parquet."""

    writer = pq.ParquetWriter(
        output / "edges.parquet",
        EDGE_SCHEMA,
        compression="zstd",
    )

    n_edges = 0
    wrote_any = False

    try:
        parquet_file = pq.ParquetFile(run_path)
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["edge_key"],
        ):
            edge_key = (
                batch.column(0)
                .to_numpy(zero_copy_only=False)
                .astype(np.int64, copy=False)
            )

            if len(edge_key) == 0:
                continue

            vertex1 = (edge_key // int(n_vertices)).astype(np.int32, copy=False)
            vertex2 = (edge_key % int(n_vertices)).astype(np.int32, copy=False)

            edge_table = pa.table(
                {
                    "edge_key": pa.array(edge_key, type=pa.int64()),
                    "vertex1": pa.array(vertex1, type=pa.int32()),
                    "vertex2": pa.array(vertex2, type=pa.int32()),
                },
                schema=EDGE_SCHEMA,
            )

            writer.write_table(edge_table)
            wrote_any = True
            n_edges += int(len(edge_key))

            del edge_key
            del vertex1
            del vertex2
            del edge_table

        if not wrote_any:
            writer.write_table(EDGE_SCHEMA.empty_table())

    finally:
        writer.close()

    return int(n_edges)


def orthorhombic_box_lengths_raw(box: np.ndarray | None) -> np.ndarray | None:
    if box is None:
        return None

    box = np.asarray(box, dtype=np.float64)

    if len(box) < 3:
        return None

    if np.any(box[:3] <= 0.0):
        return None

    if len(box) >= 6 and np.any(np.abs(box[3:6] - 90.0) > 1e-4):
        raise RedgewiseComputeError(
            "triclinic boxes are not supported by the streaming cell-list backend yet"
        )

    return box[:3]


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


def frame_index_from_ts(ts: Any, fallback: int) -> int:
    frame = getattr(ts, "frame", None)

    if frame is None:
        return fallback

    frame = int(frame)

    if frame < 0:
        return fallback

    return frame


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
    n_excluded_atom_pairs: int,
    coordinate_inference: CoordinateScaleInference,
) -> None:
    metadata = {
        "redgewise_version": __version__,
        "schema": "redgewise_sparse_undirected_per_frame_v1",
        "stride": options.stride,
        "frames_per_part": getattr(options, "frames_per_part", 1),
        "length_unit": "nm",
        "energy_unit": "kJ/mol",
        "derivative_unit": "kJ/mol/nm",
        "coordinate_unit_internal": "nm",
        "coordinate_scale_to_nm": coordinate_inference.scale_to_nm,
        "coordinate_input_unit_inferred": coordinate_inference.inferred_input_unit,
        "coordinate_unit_inference_method": coordinate_inference.method,
        "coordinate_unit_inference_n_pairs": coordinate_inference.n_pairs_sampled,
        "coordinate_unit_inference_median_distance_input": (
            coordinate_inference.median_distance_input
        ),
        "coordinate_unit_inference_median_distance_nm": (
            coordinate_inference.median_distance_nm
        ),
        "coordinate_unit_inference_p01_distance_nm": coordinate_inference.p01_distance_nm,
        "coordinate_unit_inference_p05_distance_nm": coordinate_inference.p05_distance_nm,
        "coordinate_unit_inference_p50_distance_nm": coordinate_inference.p50_distance_nm,
        "coordinate_unit_inference_p95_distance_nm": coordinate_inference.p95_distance_nm,
        "coordinate_unit_inference_too_close_fraction_nm": (
            coordinate_inference.too_close_fraction_nm
        ),
        "coordinate_unit_inference_cutoff_fraction_nm": (
            coordinate_inference.cutoff_fraction_nm
        ),
        "coordinate_unit_inference_note": coordinate_inference.note,
        "distance_backend": "streaming_orthorhombic_cell_list",
        "atom_block_size": ATOM_BLOCK_SIZE,
        "n_atoms": n_atoms,
        "n_vertices": n_vertices,
        "n_edges": n_edges,
        "n_value_rows": n_value_rows,
        "n_excluded_atom_pairs": n_excluded_atom_pairs,
        "nonbonded_exclusion_model": (
            "nrexcl_bonds_constraints_explicit_exclusions_pairs_ignored"
        ),
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