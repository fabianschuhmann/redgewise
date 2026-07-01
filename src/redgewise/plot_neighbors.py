from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq

from redgewise.analysis_values import compute_edge_analysis_summary
from redgewise.selectors import SelectorError, evaluate_vertex_selector, vertex_records_to_columns


@dataclass(frozen=True)
class VertexRecord:
    vertex_id: int
    kind: str
    label: str
    residue_name: str
    residue_id: int | None
    molecule_type: str
    molecule_instance: int | None
    atom_nr: int | None
    atom_name: str
    atom_type: str
    charge: float | None
    members: int | None


@dataclass(frozen=True)
class VertexNeighborMetric:
    vertex_id: int
    category: str
    neighbor_edges: int
    neighbor_value: float


@dataclass(frozen=True)
class ResidueNeighborRow:
    molecule_type: str
    molecule_instance: int | None
    residue_id: int
    residue_name: str
    category: str
    n_vertices: int
    n_high_res_vertices: int
    n_residue_vertices: int
    n_neighbor_edges: int
    mean_neighbor_value: float
    median_neighbor_value: float
    min_neighbor_value: float
    max_neighbor_value: float


@dataclass(frozen=True)
class NeighborPlotOutputs:
    residue_plot: Path
    residue_table: Path
    n_vertices_total: int
    n_vertices_plotted: int
    n_edges_considered: int
    n_removed_vertices: int
    n_excluded_vertices: int
    n_vertices_without_residue_id: int
    target_selector: str | None
    n_target_vertices_selected: int
    n_target_vertices_used: int
    split_excluded: bool


def run_plot_neighbors(args) -> None:
    outputs = plot_neighbors(
        input_dir=args.input,
        output=args.output,
        value_name=args.value,
        normalization=args.normalize,
        exclude_kinds=args.exclude_kind,
        exclude_resnames=args.exclude_resname,
        exclude_labels=args.exclude_label,
        exclude_vertex_ids=args.exclude_vertex_id,
        remove_kinds=args.remove_kind,
        remove_resnames=args.remove_resname,
        remove_labels=args.remove_label,
        remove_vertex_ids=args.remove_vertex_id,
        target_selector=args.target,
        neighbor_summary=args.neighbor_summary,
        split_excluded=args.split_excluded,
        min_abs_value=args.min_abs_value,
        min_abs_percentile=args.min_abs_percentile,
        molecule_delimiter_min_size=args.molecule_delimiter_min_size,
        renumber_molecule_residues=args.renumber_molecule_residues,
    )

    print("Neighbor plot written:")
    print(f"  residue plot:          {outputs.residue_plot}")
    print(f"  residue table:         {outputs.residue_table}")
    print(f"  vertices total:        {outputs.n_vertices_total}")
    print(f"  vertices plotted:      {outputs.n_vertices_plotted}")
    print(f"  edges considered:      {outputs.n_edges_considered}")
    print(f"  removed vertices:      {outputs.n_removed_vertices}")
    print(f"  excluded vertices:     {outputs.n_excluded_vertices}")
    print(f"  split excluded:        {outputs.split_excluded}")
    if outputs.target_selector:
        print(f"  target selector:       {outputs.target_selector}")
        print(f"  target selected/used:  {outputs.n_target_vertices_selected}/{outputs.n_target_vertices_used}")
    if outputs.n_vertices_without_residue_id:
        print(
            "  warning: skipped "
            f"{outputs.n_vertices_without_residue_id} plotted vertices without residue_id"
        )


def plot_neighbors(
    input_dir: Path,
    output: Path,
    value_name: str = "vdw+cl",
    normalization: str = "none",
    exclude_kinds: Iterable[str] = (),
    exclude_resnames: Iterable[str] = (),
    exclude_labels: Iterable[str] = (),
    exclude_vertex_ids: Iterable[int] = (),
    remove_kinds: Iterable[str] = (),
    remove_resnames: Iterable[str] = (),
    remove_labels: Iterable[str] = (),
    remove_vertex_ids: Iterable[int] = (),
    target_selector: str | None = None,
    neighbor_summary: str = "mean_abs",
    split_excluded: bool = False,
    min_abs_value: str | float | None = "none",
    min_abs_percentile: float = 0.05,
    molecule_delimiter_min_size: float = math.inf,
    renumber_molecule_residues: bool = False,
) -> NeighborPlotOutputs:
    input_dir = input_dir.expanduser().resolve()
    residue_plot = resolve_output_plot_path(output)
    residue_table = residue_plot.with_suffix(".tsv")

    vertices = read_vertices(input_dir / "vertices.parquet")
    n_vertices = len(vertices)

    edge_summary = compute_edge_analysis_summary(
        input_dir=input_dir,
        value_name=value_name,
        normalization=normalization,
        exclude_kinds=(),
        exclude_resnames=(),
        exclude_labels=(),
        exclude_vertex_ids=(),
        min_abs_value=min_abs_value,
        min_abs_percentile=min_abs_percentile,
        max_edges=None,
    )

    removed_mask = build_vertex_mask(
        vertices=vertices,
        kinds=remove_kinds,
        resnames=remove_resnames,
        labels=remove_labels,
        vertex_ids=remove_vertex_ids,
    )
    excluded_mask = build_vertex_mask(
        vertices=vertices,
        kinds=exclude_kinds,
        resnames=exclude_resnames,
        labels=exclude_labels,
        vertex_ids=exclude_vertex_ids,
    )

    target_mask, n_target_selected, n_target_used = resolve_target_mask(
        vertices=vertices,
        selector=target_selector,
        removed_mask=removed_mask,
    )

    metrics_by_vertex = compute_vertex_neighbor_metrics(
        edge_table=edge_summary.table,
        vertices=vertices,
        removed_mask=removed_mask,
        excluded_mask=excluded_mask,
        target_mask=target_mask,
        neighbor_summary=neighbor_summary,
        split_excluded=split_excluded,
    )

    rows, n_without_residue_id = collapse_neighbor_metrics_to_residues(
        vertices=vertices,
        metrics_by_vertex=metrics_by_vertex,
    )

    residue_plot.parent.mkdir(parents=True, exist_ok=True)
    write_residue_neighbor_table(
        path=residue_table,
        rows=rows,
        value_name=edge_summary.value_name,
        normalization=edge_summary.normalization,
        neighbor_summary=neighbor_summary,
        target_selector=target_selector,
        split_excluded=split_excluded,
    )
    write_residue_neighbor_plot(
        path=residue_plot,
        rows=rows,
        value_name=edge_summary.value_name,
        normalization=edge_summary.normalization,
        neighbor_summary=neighbor_summary,
        molecule_delimiter_min_size=molecule_delimiter_min_size,
        renumber_molecule_residues=renumber_molecule_residues,
        target_selector=target_selector,
        split_excluded=split_excluded,
    )

    plotted_vertices = {
        vertex_id for vertex_id, per_category in metrics_by_vertex.items()
        if per_category
    }

    return NeighborPlotOutputs(
        residue_plot=residue_plot,
        residue_table=residue_table,
        n_vertices_total=n_vertices,
        n_vertices_plotted=len(plotted_vertices),
        n_edges_considered=edge_summary.table.num_rows,
        n_removed_vertices=int(np.count_nonzero(removed_mask)),
        n_excluded_vertices=int(np.count_nonzero(excluded_mask)),
        n_vertices_without_residue_id=n_without_residue_id,
        target_selector=target_selector,
        n_target_vertices_selected=n_target_selected,
        n_target_vertices_used=n_target_used,
        split_excluded=split_excluded,
    )


def resolve_output_plot_path(output: Path) -> Path:
    output = output.expanduser()
    if output.suffix:
        return output.resolve()
    return (output / "neighbors.png").resolve()


def read_vertices(path: Path) -> list[VertexRecord]:
    table = pq.read_table(path)
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    n = table.num_rows

    records: list[VertexRecord] = []
    for i in range(n):
        records.append(
            VertexRecord(
                vertex_id=int(get_value(columns, "vertex_id", i, i)),
                kind=str(get_value(columns, "kind", i, "")),
                label=str(get_value(columns, "label", i, "")),
                residue_name=str(get_value(columns, "residue_name", i, "")),
                residue_id=optional_int(get_value(columns, "residue_id", i, None)),
                molecule_type=str(get_value(columns, "molecule_type", i, "")),
                molecule_instance=optional_int(get_value(columns, "molecule_instance", i, None)),
                atom_nr=optional_int(get_value(columns, "atom_nr", i, None)),
                atom_name=str(get_value(columns, "atom_name", i, "")),
                atom_type=str(get_value(columns, "atom_type", i, "")),
                charge=optional_float(get_value(columns, "charge", i, None)),
                members=optional_int(get_value(columns, "members", i, None)),
            )
        )
    return records


def get_value(columns: dict[str, list[object]], name: str, index: int, default: object) -> object:
    if name not in columns:
        return default
    value = columns[name][index]
    return default if value is None else value


def optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def build_vertex_mask(
    vertices: list[VertexRecord],
    kinds: Iterable[str] = (),
    resnames: Iterable[str] = (),
    labels: Iterable[str] = (),
    vertex_ids: Iterable[int] = (),
) -> np.ndarray:
    kinds_set = {str(value).strip() for value in kinds if str(value).strip()}
    resnames_set = {str(value).strip().upper() for value in resnames if str(value).strip()}
    labels_set = {str(value).strip() for value in labels if str(value).strip()}
    vertex_ids_set = {int(value) for value in vertex_ids}

    mask = np.zeros(len(vertices), dtype=bool)
    for vertex in vertices:
        vertex_id = vertex.vertex_id
        selected = False
        if kinds_set and vertex.kind in kinds_set:
            selected = True
        if resnames_set and vertex.residue_name.upper() in resnames_set:
            selected = True
        if labels_set and vertex.label in labels_set:
            selected = True
        if vertex_ids_set and vertex_id in vertex_ids_set:
            selected = True
        if 0 <= vertex_id < len(mask):
            mask[vertex_id] = selected
    return mask


def resolve_target_mask(
    vertices: list[VertexRecord],
    selector: str | None,
    removed_mask: np.ndarray,
) -> tuple[np.ndarray | None, int, int]:
    if selector is None or not selector.strip():
        return None, 0, 0

    columns = vertex_records_to_columns(vertices)
    try:
        selected = evaluate_vertex_selector(selector, columns, n_rows=len(vertices))
    except SelectorError as exc:
        raise ValueError(f"invalid --target selector: {exc}") from exc

    selected = np.asarray(selected, dtype=bool)
    n_selected = int(np.count_nonzero(selected))
    if n_selected == 0:
        raise ValueError(f"--target selector matched no vertices: {selector!r}")

    used = selected & ~removed_mask
    n_used = int(np.count_nonzero(used))
    if n_used == 0:
        raise ValueError(
            "--target selector matched vertices, but all were removed by --remove-* filters"
        )
    return used, n_selected, n_used


def compute_vertex_neighbor_metrics(
    edge_table,
    vertices: list[VertexRecord],
    removed_mask: np.ndarray,
    excluded_mask: np.ndarray,
    target_mask: np.ndarray | None,
    neighbor_summary: str,
    split_excluded: bool,
) -> dict[int, dict[str, VertexNeighborMetric]]:
    mode = canonical_neighbor_summary(neighbor_summary)
    n_vertices = len(vertices)
    source_allowed = ~removed_mask & ~excluded_mask

    sums: dict[tuple[int, str], float] = {}
    counts: dict[tuple[int, str], int] = {}
    values_for_median: dict[tuple[int, str], list[float]] = {}

    vertex1 = edge_table.column("vertex1").to_numpy(zero_copy_only=False).astype(np.int64)
    vertex2 = edge_table.column("vertex2").to_numpy(zero_copy_only=False).astype(np.int64)
    values = edge_table.column("value").to_numpy(zero_copy_only=False).astype(np.float64)

    for v1, v2, value in zip(vertex1, vertex2, values):
        v1 = int(v1)
        v2 = int(v2)
        value = float(value)
        if not math.isfinite(value):
            continue
        if v1 < 0 or v2 < 0 or v1 >= n_vertices or v2 >= n_vertices:
            continue
        if removed_mask[v1] or removed_mask[v2]:
            continue
        add_neighbor_contribution(
            source=v1,
            neighbor=v2,
            value=value,
            source_allowed=source_allowed,
            excluded_mask=excluded_mask,
            target_mask=target_mask,
            split_excluded=split_excluded,
            summary_mode=mode,
            sums=sums,
            counts=counts,
            values_for_median=values_for_median,
        )
        add_neighbor_contribution(
            source=v2,
            neighbor=v1,
            value=value,
            source_allowed=source_allowed,
            excluded_mask=excluded_mask,
            target_mask=target_mask,
            split_excluded=split_excluded,
            summary_mode=mode,
            sums=sums,
            counts=counts,
            values_for_median=values_for_median,
        )

    result: dict[int, dict[str, VertexNeighborMetric]] = {}
    all_keys = sorted(counts)
    for vertex_id, category in all_keys:
        count = counts[(vertex_id, category)]
        if count <= 0:
            continue
        if mode in {"mean", "mean_abs"}:
            metric_value = sums[(vertex_id, category)] / float(count)
        elif mode in {"sum", "sum_abs"}:
            metric_value = sums[(vertex_id, category)]
        elif mode in {"median", "median_abs"}:
            metric_value = float(np.median(np.asarray(values_for_median[(vertex_id, category)], dtype=np.float64)))
        else:  # pragma: no cover
            raise ValueError(f"unknown neighbor summary: {mode}")
        result.setdefault(vertex_id, {})[category] = VertexNeighborMetric(
            vertex_id=vertex_id,
            category=category,
            neighbor_edges=count,
            neighbor_value=float(metric_value),
        )
    return result


def add_neighbor_contribution(
    source: int,
    neighbor: int,
    value: float,
    source_allowed: np.ndarray,
    excluded_mask: np.ndarray,
    target_mask: np.ndarray | None,
    split_excluded: bool,
    summary_mode: str,
    sums: dict[tuple[int, str], float],
    counts: dict[tuple[int, str], int],
    values_for_median: dict[tuple[int, str], list[float]],
) -> None:
    if not source_allowed[source]:
        return
    if target_mask is not None and not target_mask[neighbor]:
        return

    if split_excluded:
        category = "excluded" if excluded_mask[neighbor] else "non_excluded"
    else:
        category = "all"

    if summary_mode.endswith("_abs"):
        contribution = abs(value)
    else:
        contribution = value

    key = (source, category)
    sums[key] = sums.get(key, 0.0) + float(contribution)
    counts[key] = counts.get(key, 0) + 1
    if summary_mode in {"median", "median_abs"}:
        values_for_median.setdefault(key, []).append(float(contribution))


def canonical_neighbor_summary(summary: str) -> str:
    key = summary.strip().lower()
    aliases = {
        "average": "mean",
        "average_abs": "mean_abs",
        "avg": "mean",
        "avg_abs": "mean_abs",
    }
    key = aliases.get(key, key)
    allowed = {"mean", "mean_abs", "sum", "sum_abs", "median", "median_abs"}
    if key not in allowed:
        raise ValueError(
            f"unknown neighbor summary {summary!r}; expected one of: "
            f"{', '.join(sorted(allowed))}"
        )
    return key


def collapse_neighbor_metrics_to_residues(
    vertices: list[VertexRecord],
    metrics_by_vertex: dict[int, dict[str, VertexNeighborMetric]],
) -> tuple[list[ResidueNeighborRow], int]:
    grouped: dict[tuple[str, int | None, int, str], list[tuple[VertexRecord, VertexNeighborMetric]]] = {}
    n_without_residue_id = 0

    for vertex in vertices:
        per_category = metrics_by_vertex.get(vertex.vertex_id)
        if not per_category:
            continue
        if vertex.residue_id is None:
            n_without_residue_id += 1
            continue
        for category, metric in per_category.items():
            if not math.isfinite(metric.neighbor_value):
                continue
            key = (vertex.molecule_type, vertex.molecule_instance, int(vertex.residue_id), category)
            grouped.setdefault(key, []).append((vertex, metric))

    rows: list[ResidueNeighborRow] = []
    for molecule_type, molecule_instance, residue_id, category in sorted(
        grouped,
        key=lambda key: (key[0], 10**18 if key[1] is None else key[1], key[2], key[3]),
    ):
        items = grouped[(molecule_type, molecule_instance, residue_id, category)]
        values = np.asarray([item[1].neighbor_value for item in items], dtype=np.float64)
        residue_name = first_nonempty([item[0].residue_name for item in items])
        rows.append(
            ResidueNeighborRow(
                molecule_type=molecule_type,
                molecule_instance=molecule_instance,
                residue_id=int(residue_id),
                residue_name=residue_name,
                category=category,
                n_vertices=len(items),
                n_high_res_vertices=sum(1 for vertex, _ in items if vertex.kind == "atom"),
                n_residue_vertices=sum(1 for vertex, _ in items if vertex.kind == "residue"),
                n_neighbor_edges=sum(metric.neighbor_edges for _, metric in items),
                mean_neighbor_value=float(np.mean(values)),
                median_neighbor_value=float(np.median(values)),
                min_neighbor_value=float(np.min(values)),
                max_neighbor_value=float(np.max(values)),
            )
        )
    return rows, n_without_residue_id


def first_nonempty(values: Iterable[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


def write_residue_neighbor_table(
    path: Path,
    rows: list[ResidueNeighborRow],
    value_name: str,
    normalization: str,
    neighbor_summary: str,
    target_selector: str | None,
    split_excluded: bool,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "molecule_type",
                "molecule_instance",
                "residue_id",
                "residue_name",
                "category",
                "n_vertices",
                "n_high_res_vertices",
                "n_residue_vertices",
                "n_neighbor_edges",
                "mean_neighbor_value",
                "median_neighbor_value",
                "min_neighbor_value",
                "max_neighbor_value",
                "value_name",
                "normalization",
                "neighbor_summary",
                "target_selector",
                "split_excluded",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.molecule_type,
                    "" if row.molecule_instance is None else row.molecule_instance,
                    row.residue_id,
                    row.residue_name,
                    row.category,
                    row.n_vertices,
                    row.n_high_res_vertices,
                    row.n_residue_vertices,
                    row.n_neighbor_edges,
                    format_float(row.mean_neighbor_value),
                    format_float(row.median_neighbor_value),
                    format_float(row.min_neighbor_value),
                    format_float(row.max_neighbor_value),
                    value_name,
                    normalization,
                    neighbor_summary,
                    "" if target_selector is None else target_selector,
                    split_excluded,
                ]
            )


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.15g}"


def write_residue_neighbor_plot(
    path: Path,
    rows: list[ResidueNeighborRow],
    value_name: str,
    normalization: str,
    neighbor_summary: str,
    molecule_delimiter_min_size: float = math.inf,
    renumber_molecule_residues: bool = False,
    target_selector: str | None = None,
    split_excluded: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    base_rows = sorted_unique_residue_rows(rows)
    x_plot, x_labels, jump_markers = compressed_residue_axis(
        base_rows,
        max_missing_residues=2,
        renumber_molecule_residues=renumber_molecule_residues,
    )
    x_by_key = {
        residue_key(row): float(x_plot[index])
        for index, row in enumerate(base_rows)
    }

    fig, ax = plt.subplots(figsize=(10, 4.5))
    categories = categories_for_plot(rows, split_excluded=split_excluded)
    offsets = category_offsets(categories)

    for category in categories:
        category_rows = [row for row in rows if row.category == category]
        category_rows.sort(key=residue_sort_key)
        x = np.asarray([x_by_key[residue_key(row)] + offsets.get(category, 0.0) for row in category_rows], dtype=np.float64)
        y = np.asarray([row.mean_neighbor_value for row in category_rows], dtype=np.float64)
        if len(x) == 0:
            continue
        plot_connected_neighbor_blocks(ax, category_rows, x, y, label=None if len(categories) == 1 else category)
        ax.scatter(x, y, s=14, zorder=3, label=None if has_line_for_category(category_rows) else category)

    if len(x_plot) > 0:
        add_residue_axis_jump_markers(ax, jump_markers)
        add_molecule_delimiters(ax, base_rows, x_plot, molecule_delimiter_min_size)
        set_sparse_residue_ticks(ax, x_plot, x_labels)

    if len(categories) > 1:
        ax.legend(frameon=False, fontsize=8)

    if renumber_molecule_residues:
        ax.set_xlabel("Residue index within molecule")
    else:
        ax.set_xlabel("Residue ID")
    ax.set_ylabel(f"Neighbor {neighbor_summary} value")
    target_suffix = "" if not target_selector else f"; target={target_selector}"
    split_suffix = "; split excluded" if split_excluded else ""
    ax.set_title(
        f"Direct-neighbor profile ({value_name})"
    )
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def sorted_unique_residue_rows(rows: list[ResidueNeighborRow]) -> list[ResidueNeighborRow]:
    first_by_key: dict[tuple[str, int | None, int], ResidueNeighborRow] = {}
    for row in sorted(rows, key=residue_sort_key):
        first_by_key.setdefault(residue_key(row), row)
    return [first_by_key[key] for key in sorted(first_by_key, key=lambda key: (key[0], 10**18 if key[1] is None else key[1], key[2]))]


def residue_key(row: ResidueNeighborRow) -> tuple[str, int | None, int]:
    return row.molecule_type, row.molecule_instance, row.residue_id


def residue_sort_key(row: ResidueNeighborRow) -> tuple[str, int, int, str]:
    return (
        row.molecule_type,
        10**18 if row.molecule_instance is None else row.molecule_instance,
        row.residue_id,
        row.category,
    )


def categories_for_plot(rows: list[ResidueNeighborRow], split_excluded: bool) -> list[str]:
    available = {row.category for row in rows}
    if split_excluded:
        return [category for category in ("non_excluded", "excluded") if category in available]
    return ["all"] if "all" in available else sorted(available)


def category_offsets(categories: list[str]) -> dict[str, float]:
    if len(categories) <= 1:
        return {category: 0.0 for category in categories}
    spread = 0.24
    center = (len(categories) - 1) / 2.0
    return {
        category: (index - center) * spread
        for index, category in enumerate(categories)
    }


def has_line_for_category(rows: list[ResidueNeighborRow]) -> bool:
    blocks = list(iter_molecule_blocks(rows))
    return any(stop - start >= 2 for start, stop in blocks)


def compressed_residue_axis(
    rows: list[ResidueNeighborRow],
    max_missing_residues: int,
    renumber_molecule_residues: bool = False,
) -> tuple[np.ndarray, list[str], list[tuple[float, int, int]]]:
    x_values: list[float] = []
    x_labels: list[str] = []
    jump_markers: list[tuple[float, int, int]] = []

    previous_residue_id: int | None = None
    previous_molecule_key: tuple[str, int | None] | None = None
    current_x = -1.0
    residue_index_in_molecule = 0

    for row in rows:
        current_molecule_key = molecule_block_key(row)
        starts_new_molecule = (
            previous_molecule_key is not None
            and current_molecule_key != previous_molecule_key
        )
        if starts_new_molecule:
            residue_index_in_molecule = 0

        gap = None
        if (
            previous_residue_id is not None
            and not starts_new_molecule
            and row.residue_id > previous_residue_id
        ):
            gap = row.residue_id - previous_residue_id - 1

        if gap is not None and gap > max_missing_residues:
            current_x += 2.0
            jump_markers.append((current_x - 1.0, previous_residue_id, row.residue_id))
        else:
            current_x += 1.0

        residue_index_in_molecule += 1
        x_values.append(current_x)
        x_labels.append(str(residue_index_in_molecule if renumber_molecule_residues else row.residue_id))
        previous_residue_id = row.residue_id
        previous_molecule_key = current_molecule_key

    return np.asarray(x_values, dtype=np.float64), x_labels, jump_markers


def molecule_block_key(row: ResidueNeighborRow) -> tuple[str, int | None]:
    return row.molecule_type, row.molecule_instance


def iter_molecule_blocks(rows: list[ResidueNeighborRow]) -> Iterable[tuple[int, int]]:
    if not rows:
        return
    rows_sorted = sorted(rows, key=residue_sort_key)
    start = 0
    previous = molecule_block_key(rows_sorted[0])
    for index, row in enumerate(rows_sorted[1:], start=1):
        current = molecule_block_key(row)
        if current != previous:
            yield start, index
            start = index
            previous = current
    yield start, len(rows_sorted)


def plot_connected_neighbor_blocks(ax, rows: list[ResidueNeighborRow], x: np.ndarray, y: np.ndarray, label: str | None) -> None:
    rows_sorted = sorted(rows, key=residue_sort_key)
    if len(rows_sorted) != len(x):
        return
    label_used = False
    for start, stop in iter_molecule_blocks(rows_sorted):
        if stop - start < 2:
            continue
        ax.plot(
            x[start:stop],
            y[start:stop],
            linewidth=1.0,
            alpha=0.65,
            zorder=2,
            label=label if not label_used else None,
        )
        label_used = True


def add_molecule_delimiters(
    ax,
    rows: list[ResidueNeighborRow],
    x_plot: np.ndarray,
    min_size: float,
) -> None:
    if not np.isfinite(min_size):
        return

    rows = sorted(rows, key=residue_sort_key)
    blocks = list(iter_molecule_blocks(rows))
    for block_index in range(len(blocks) - 1):
        left_start, left_stop = blocks[block_index]
        right_start, right_stop = blocks[block_index + 1]
        left_size = left_stop - left_start
        right_size = right_stop - right_start
        if left_size < min_size and right_size < min_size:
            continue
        ax.axvline(
            float(x_plot[right_start]) - 0.5,
            linewidth=0.8,
            linestyle="--",
            alpha=0.45,
            zorder=1,
        )


def add_residue_axis_jump_markers(ax, jump_markers: list[tuple[float, int, int]]) -> None:
    if not jump_markers:
        return

    ymin, ymax = ax.get_ylim()
    y_text = ymin + 0.03 * (ymax - ymin)
    for x, left_resid, right_resid in jump_markers:
        ax.axvline(x, linewidth=0.8, linestyle="--", alpha=0.55, zorder=1)
        ax.text(
            x,
            y_text,
            f"// {left_resid}→{right_resid}",
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=7,
        )


def set_sparse_residue_ticks(ax, x_plot: np.ndarray, x_labels: list[str]) -> None:
    if len(x_plot) == 0:
        return
    tick_step = max(1, int(math.ceil(len(x_plot) / 80)))
    tick_indices = list(range(0, len(x_plot), tick_step))
    if len(x_plot) - 1 not in tick_indices:
        tick_indices.append(len(x_plot) - 1)
    ax.set_xticks([float(x_plot[index]) for index in tick_indices])
    ax.set_xticklabels([x_labels[index] for index in tick_indices], rotation=90, fontsize=7)


def clean_axes(ax) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3.0)
    ax.spines["bottom"].set_linewidth(3.0)
    ax.tick_params(axis="both", width=3.0, length=7.0)
