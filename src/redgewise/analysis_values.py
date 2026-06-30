from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


VALUE_COLUMNS = {
    "vdw": ("vdw",),
    "cl": ("coulomb",),
    "coulomb": ("coulomb",),
    "vdw+cl": ("vdw", "coulomb"),
    "vdw+coulomb": ("vdw", "coulomb"),
    "dvdw": ("vdw_dif",),
    "dcl": ("coulomb_dif",),
    "dcoulomb": ("coulomb_dif",),
    "dvdw+dcl": ("vdw_dif", "coulomb_dif"),
    "dvdw+dcoulomb": ("vdw_dif", "coulomb_dif"),
}

NORMALIZATION_MODES = (
    "none",
    "per_atom_pair",
    "per_vertex_member_sqrt",
    "per_vertex_member_product",
    "per_coarse_member_sqrt",
    "per_coarse_member_product",
)

COARSE_VERTEX_KINDS = {"low_res", "bundle"}


@dataclass(frozen=True)
class EdgeAnalysisSummary:
    table: pa.Table
    value_name: str
    normalization: str
    n_frames: int
    n_edges_before_filter: int
    n_edges_after_exclusion: int
    n_edges_after_threshold: int
    n_edges_final: int
    min_abs_value_used: float
    min_abs_value_mode: str


@dataclass(frozen=True)
class AggregatedEdgeValues:
    edge_value_sum: dict[int, float]
    atom_pair_count_sum: dict[int, int]
    n_frames: int


def compute_edge_analysis_summary(
    input_dir: Path,
    value_name: str,
    normalization: str = "none",
    exclude_kinds: Iterable[str] = (),
    exclude_resnames: Iterable[str] = (),
    exclude_labels: Iterable[str] = (),
    exclude_vertex_ids: Iterable[int] = (),
    min_abs_value: str | float | None = "auto",
    min_abs_percentile: float = 0.05,
    max_edges: int | None = None,
) -> EdgeAnalysisSummary:
    input_dir = input_dir.expanduser().resolve()
    validate_redgewise_output(input_dir)

    value_name = canonical_value_name(value_name)
    normalization = canonical_normalization(normalization)

    vertices = pq.read_table(input_dir / "vertices.parquet")
    edges = pq.read_table(input_dir / "edges.parquet")
    vertex_members = pq.read_table(input_dir / "vertex_members.parquet")

    aggregated = aggregate_edge_values(
        input_dir=input_dir,
        value_name=value_name,
    )

    table = build_edge_analysis_table(
        vertices=vertices,
        edges=edges,
        vertex_members=vertex_members,
        aggregated=aggregated,
        normalization=normalization,
    )

    n_edges_before_filter = table.num_rows

    table = apply_endpoint_exclusions(
        table=table,
        exclude_kinds=exclude_kinds,
        exclude_resnames=exclude_resnames,
        exclude_labels=exclude_labels,
        exclude_vertex_ids=exclude_vertex_ids,
    )

    n_edges_after_exclusion = table.num_rows

    threshold, threshold_mode = resolve_min_abs_value(
        table=table,
        min_abs_value=min_abs_value,
        min_abs_percentile=min_abs_percentile,
    )

    table = apply_min_abs_value(table=table, min_abs_value=threshold)
    n_edges_after_threshold = table.num_rows

    table = apply_max_edges(table=table, max_edges=max_edges)
    n_edges_final = table.num_rows

    return EdgeAnalysisSummary(
        table=table,
        value_name=value_name,
        normalization=normalization,
        n_frames=aggregated.n_frames,
        n_edges_before_filter=n_edges_before_filter,
        n_edges_after_exclusion=n_edges_after_exclusion,
        n_edges_after_threshold=n_edges_after_threshold,
        n_edges_final=n_edges_final,
        min_abs_value_used=threshold,
        min_abs_value_mode=threshold_mode,
    )


def validate_redgewise_output(input_dir: Path) -> None:
    required_files = [
        "vertices.parquet",
        "vertex_members.parquet",
        "edges.parquet",
        "metadata.json",
    ]

    for name in required_files:
        path = input_dir / name
        if not path.exists():
            raise FileNotFoundError(f"missing redgewise output file: {path}")

    values_dir = input_dir / "values"
    if not values_dir.exists():
        raise FileNotFoundError(f"missing redgewise values directory: {values_dir}")

    if not any(values_dir.glob("*.parquet")):
        raise FileNotFoundError(f"no parquet value files found in: {values_dir}")


def canonical_value_name(value_name: str) -> str:
    key = value_name.strip().lower()

    if key not in VALUE_COLUMNS:
        raise ValueError(
            f"unknown value {value_name!r}; expected one of: "
            f"{', '.join(sorted(VALUE_COLUMNS))}"
        )

    aliases = {
        "coulomb": "cl",
        "vdw+coulomb": "vdw+cl",
        "dcoulomb": "dcl",
        "dvdw+dcoulomb": "dvdw+dcl",
    }

    return aliases.get(key, key)


def canonical_normalization(normalization: str) -> str:
    key = normalization.strip().lower()

    if key == "raw":
        key = "none"

    if key not in NORMALIZATION_MODES:
        raise ValueError(
            f"unknown normalization {normalization!r}; expected one of: "
            f"{', '.join(NORMALIZATION_MODES)}"
        )

    return key


def aggregate_edge_values(input_dir: Path, value_name: str) -> AggregatedEdgeValues:
    columns = VALUE_COLUMNS[value_name]

    edge_value_sum: dict[int, float] = {}
    atom_pair_count_sum: dict[int, int] = {}
    frames: set[int] = set()

    for path in sorted((input_dir / "values").glob("*.parquet")):
        table = pq.read_table(path, columns=["frame", "edge_key", *columns, "n_atom_pairs"])

        frame_array = table.column("frame").to_numpy(zero_copy_only=False)
        edge_key_array = table.column("edge_key").to_numpy(zero_copy_only=False)
        n_atom_pairs_array = table.column("n_atom_pairs").to_numpy(zero_copy_only=False)

        frames.update(int(frame) for frame in np.unique(frame_array))

        value_array = np.zeros(len(edge_key_array), dtype=np.float64)
        for column in columns:
            value_array += table.column(column).to_numpy(zero_copy_only=False)

        unique_edge_keys, inverse = np.unique(edge_key_array, return_inverse=True)
        value_sums = np.bincount(inverse, weights=value_array)
        pair_count_sums = np.bincount(inverse, weights=n_atom_pairs_array).astype(np.int64)

        for index, edge_key in enumerate(unique_edge_keys):
            key = int(edge_key)
            edge_value_sum[key] = edge_value_sum.get(key, 0.0) + float(value_sums[index])
            atom_pair_count_sum[key] = atom_pair_count_sum.get(key, 0) + int(pair_count_sums[index])

    n_frames = len(frames)

    if n_frames == 0:
        raise ValueError("no frames found in values parquet files")

    return AggregatedEdgeValues(
        edge_value_sum=edge_value_sum,
        atom_pair_count_sum=atom_pair_count_sum,
        n_frames=n_frames,
    )


def build_edge_analysis_table(
    vertices: pa.Table,
    edges: pa.Table,
    vertex_members: pa.Table,
    aggregated: AggregatedEdgeValues,
    normalization: str,
) -> pa.Table:
    edge_key = edges.column("edge_key").to_numpy(zero_copy_only=False).astype(np.int64)
    vertex1 = edges.column("vertex1").to_numpy(zero_copy_only=False).astype(np.int32)
    vertex2 = edges.column("vertex2").to_numpy(zero_copy_only=False).astype(np.int32)

    value_sum = np.array(
        [aggregated.edge_value_sum.get(int(key), 0.0) for key in edge_key],
        dtype=np.float64,
    )
    atom_pair_count_sum = np.array(
        [aggregated.atom_pair_count_sum.get(int(key), 0) for key in edge_key],
        dtype=np.int64,
    )

    value_raw = value_sum / float(aggregated.n_frames)
    n_atom_pairs_mean = atom_pair_count_sum.astype(np.float64) / float(aggregated.n_frames)

    member_counts = compute_vertex_member_counts(
        n_vertices=vertices.num_rows,
        vertex_members=vertex_members,
    )

    labels = get_string_column(vertices, "label")
    kinds = get_string_column(vertices, "kind")
    residue_names = get_string_column(vertices, "residue_name")

    n_members1 = member_counts[vertex1].astype(np.int32)
    n_members2 = member_counts[vertex2].astype(np.int32)
    kind1 = [kinds[int(vertex_id)] for vertex_id in vertex1]
    kind2 = [kinds[int(vertex_id)] for vertex_id in vertex2]
    label1 = [labels[int(vertex_id)] for vertex_id in vertex1]
    label2 = [labels[int(vertex_id)] for vertex_id in vertex2]
    resname1 = [residue_names[int(vertex_id)] for vertex_id in vertex1]
    resname2 = [residue_names[int(vertex_id)] for vertex_id in vertex2]

    value = normalize_values(
        value_raw=value_raw,
        atom_pair_count_sum=atom_pair_count_sum,
        n_members1=n_members1,
        n_members2=n_members2,
        kind1=kind1,
        kind2=kind2,
        normalization=normalization,
    )

    return pa.table(
        {
            "edge_key": pa.array(edge_key, type=pa.int64()),
            "vertex1": pa.array(vertex1, type=pa.int32()),
            "vertex2": pa.array(vertex2, type=pa.int32()),
            "value_raw": pa.array(value_raw.astype(np.float64), type=pa.float64()),
            "value": pa.array(value.astype(np.float64), type=pa.float64()),
            "abs_value": pa.array(np.abs(value).astype(np.float64), type=pa.float64()),
            "n_atom_pairs_sum": pa.array(atom_pair_count_sum, type=pa.int64()),
            "n_atom_pairs_mean": pa.array(n_atom_pairs_mean.astype(np.float64), type=pa.float64()),
            "n_members1": pa.array(n_members1, type=pa.int32()),
            "n_members2": pa.array(n_members2, type=pa.int32()),
            "kind1": pa.array(kind1, type=pa.string()),
            "kind2": pa.array(kind2, type=pa.string()),
            "label1": pa.array(label1, type=pa.string()),
            "label2": pa.array(label2, type=pa.string()),
            "resname1": pa.array(resname1, type=pa.string()),
            "resname2": pa.array(resname2, type=pa.string()),
        }
    )


def compute_vertex_member_counts(n_vertices: int, vertex_members: pa.Table) -> np.ndarray:
    vertex_ids = vertex_members.column("vertex_id").to_numpy(zero_copy_only=False).astype(np.int64)

    if len(vertex_ids) == 0:
        raise ValueError("vertex_members.parquet contains no rows")

    if np.min(vertex_ids) < 0 or np.max(vertex_ids) >= n_vertices:
        raise ValueError("vertex_members.parquet contains invalid vertex_id values")

    counts = np.bincount(vertex_ids, minlength=n_vertices).astype(np.int32)

    if np.any(counts <= 0):
        missing = np.where(counts <= 0)[0]
        raise ValueError(
            "vertex_members.parquet has vertices without members; first missing "
            f"vertex_id={int(missing[0])}"
        )

    return counts


def get_string_column(table: pa.Table, name: str) -> list[str]:
    if name not in table.column_names:
        return [""] * table.num_rows

    return ["" if value is None else str(value) for value in table.column(name).to_pylist()]


def normalize_values(
    value_raw: np.ndarray,
    atom_pair_count_sum: np.ndarray,
    n_members1: np.ndarray,
    n_members2: np.ndarray,
    kind1: list[str],
    kind2: list[str],
    normalization: str,
) -> np.ndarray:
    if normalization == "none":
        return value_raw.copy()

    if normalization == "per_atom_pair":
        denominator = atom_pair_count_sum.astype(np.float64)
        return divide_safely(value_raw, denominator)

    if normalization == "per_vertex_member_sqrt":
        denominator = np.sqrt(n_members1.astype(np.float64) * n_members2.astype(np.float64))
        return divide_safely(value_raw, denominator)

    if normalization == "per_vertex_member_product":
        denominator = n_members1.astype(np.float64) * n_members2.astype(np.float64)
        return divide_safely(value_raw, denominator)

    if normalization in {"per_coarse_member_sqrt", "per_coarse_member_product"}:
        factor1 = coarse_member_factor(n_members1, kind1)
        factor2 = coarse_member_factor(n_members2, kind2)
        denominator = factor1 * factor2

        if normalization == "per_coarse_member_sqrt":
            denominator = np.sqrt(denominator)

        return divide_safely(value_raw, denominator)

    raise ValueError(f"unknown normalization: {normalization}")


def coarse_member_factor(n_members: np.ndarray, kinds: list[str]) -> np.ndarray:
    factor = np.ones(len(n_members), dtype=np.float64)

    for index, kind in enumerate(kinds):
        if kind in COARSE_VERTEX_KINDS:
            factor[index] = float(n_members[index])

    return factor


def divide_safely(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros(len(numerator), dtype=np.float64)
    keep = denominator > 0.0
    result[keep] = numerator[keep] / denominator[keep]
    return result


def apply_endpoint_exclusions(
    table: pa.Table,
    exclude_kinds: Iterable[str],
    exclude_resnames: Iterable[str],
    exclude_labels: Iterable[str],
    exclude_vertex_ids: Iterable[int],
) -> pa.Table:
    keep = np.ones(table.num_rows, dtype=bool)

    kinds = {value.strip() for value in exclude_kinds if str(value).strip()}
    if kinds:
        kind1 = np.array(table.column("kind1").to_pylist(), dtype=object)
        kind2 = np.array(table.column("kind2").to_pylist(), dtype=object)
        keep &= ~(np.isin(kind1, list(kinds)) | np.isin(kind2, list(kinds)))

    resnames = {str(value).strip().upper() for value in exclude_resnames if str(value).strip()}
    if resnames:
        resname1 = np.array([str(value).upper() for value in table.column("resname1").to_pylist()], dtype=object)
        resname2 = np.array([str(value).upper() for value in table.column("resname2").to_pylist()], dtype=object)
        keep &= ~(np.isin(resname1, list(resnames)) | np.isin(resname2, list(resnames)))

    labels = {str(value).strip() for value in exclude_labels if str(value).strip()}
    if labels:
        label1 = np.array(table.column("label1").to_pylist(), dtype=object)
        label2 = np.array(table.column("label2").to_pylist(), dtype=object)
        keep &= ~(np.isin(label1, list(labels)) | np.isin(label2, list(labels)))

    vertex_ids = {int(value) for value in exclude_vertex_ids}
    if vertex_ids:
        vertex1 = table.column("vertex1").to_numpy(zero_copy_only=False)
        vertex2 = table.column("vertex2").to_numpy(zero_copy_only=False)
        keep &= ~(np.isin(vertex1, list(vertex_ids)) | np.isin(vertex2, list(vertex_ids)))

    return table.filter(pa.array(keep))


def resolve_min_abs_value(
    table: pa.Table,
    min_abs_value: str | float | None,
    min_abs_percentile: float,
) -> tuple[float, str]:
    if table.num_rows == 0:
        return 0.0, "empty"

    if min_abs_value is None:
        return 0.0, "none"

    if isinstance(min_abs_value, str):
        text = min_abs_value.strip().lower()
        if text in {"none", "off", "0", "0.0"}:
            return 0.0, "none"
        if text != "auto":
            try:
                return float(text), "explicit"
            except ValueError as exc:
                raise ValueError(
                    "--min-abs-value must be a number, 'auto', or 'none'"
                ) from exc
    else:
        return float(min_abs_value), "explicit"

    if min_abs_percentile < 0.0 or min_abs_percentile > 100.0:
        raise ValueError("--min-abs-percentile must be between 0 and 100")

    abs_values = table.column("abs_value").to_numpy(zero_copy_only=False).astype(np.float64)
    finite_nonzero = abs_values[np.isfinite(abs_values) & (abs_values > 0.0)]

    if len(finite_nonzero) == 0:
        return 0.0, "auto_empty"

    threshold = float(np.percentile(finite_nonzero, min_abs_percentile))
    return threshold, f"auto_p{min_abs_percentile:g}"


def apply_min_abs_value(table: pa.Table, min_abs_value: float) -> pa.Table:
    if table.num_rows == 0 or min_abs_value <= 0.0:
        return table

    abs_values = table.column("abs_value").to_numpy(zero_copy_only=False)
    keep = np.isfinite(abs_values) & (abs_values >= min_abs_value)
    return table.filter(pa.array(keep))


def apply_max_edges(table: pa.Table, max_edges: int | None) -> pa.Table:
    if max_edges is None:
        return table

    if max_edges < 1:
        raise ValueError("--max-edges must be >= 1")

    if table.num_rows <= max_edges:
        return table

    abs_values = table.column("abs_value").to_numpy(zero_copy_only=False)
    order = np.argsort(abs_values)[::-1][:max_edges]
    return table.take(pa.array(order.astype(np.int64)))
