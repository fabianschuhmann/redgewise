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


@dataclass(frozen=True)
class VertexPathMetrics:
    vertex_id: int
    reachable_vertices: int
    mean_shortest_path: float
    median_shortest_path: float
    min_shortest_path: float
    max_shortest_path: float


@dataclass(frozen=True)
class ShortestPathOutputs:
    residue_plot: Path
    high_res_plot: Path | None
    residue_table: Path
    high_res_table: Path | None
    n_vertices_used: int
    n_edges_used: int
    n_isolated_vertices: int
    n_vertices_without_residue_id: int
    n_high_res_vertices_plotted: int
    n_unique_vertex_means: int
    mean_min: float
    mean_median: float
    mean_max: float
    target_selector: str | None = None
    n_target_vertices_selected: int = 0
    n_target_vertices_used: int = 0


def run_plot_shortest_path(args) -> None:
    outputs = plot_shortest_path(
        input_dir=args.input,
        output=args.output,
        value_name=args.value,
        normalization=args.normalize,
        exclude_kinds=args.exclude_kind,
        exclude_resnames=args.exclude_resname,
        exclude_labels=args.exclude_label,
        exclude_vertex_ids=args.exclude_vertex_id,
        min_abs_value=args.min_abs_value,
        min_abs_percentile=args.min_abs_percentile,
        molecule_delimiter_min_size=args.molecule_delimiter_min_size,
        renumber_molecule_residues=args.renumber_molecule_residues,
        target_selector=args.target,
    )

    print("Shortest-path plot written:")
    print(f"  residue plot:    {outputs.residue_plot}")
    if outputs.high_res_plot is not None:
        print(f"  high-res plot:   {outputs.high_res_plot}")
    else:
        print("  high-res plot:   not written; no high-resolution atom vertices with finite values")
    print(f"  residue table:   {outputs.residue_table}")
    if outputs.high_res_table is not None:
        print(f"  high-res table:  {outputs.high_res_table}")

    print("\nGraph:")
    print(f"  vertices used:                 {outputs.n_vertices_used}")
    print(f"  edges used:                    {outputs.n_edges_used}")
    print(f"  isolated vertices:             {outputs.n_isolated_vertices}")
    print(f"  vertices without residue_id:   {outputs.n_vertices_without_residue_id}")
    print(f"  high-res vertices plotted:     {outputs.n_high_res_vertices_plotted}")

    if outputs.target_selector:
        print("\nTarget selector:")
        print(f"  selector:                      {outputs.target_selector}")
        print(f"  selected vertices:             {outputs.n_target_vertices_selected}")
        print(f"  used target vertices:          {outputs.n_target_vertices_used}")

    print("\nShortest-path source means:")
    print(f"  unique finite means:           {outputs.n_unique_vertex_means}")
    print(f"  min:                           {format_float(outputs.mean_min)}")
    print(f"  median:                        {format_float(outputs.mean_median)}")
    print(f"  max:                           {format_float(outputs.mean_max)}")


def plot_shortest_path(
    input_dir: Path,
    output: Path,
    value_name: str = "vdw+cl",
    normalization: str = "none",
    exclude_kinds: Iterable[str] = (),
    exclude_resnames: Iterable[str] = (),
    exclude_labels: Iterable[str] = (),
    exclude_vertex_ids: Iterable[int] = (),
    min_abs_value: str | float | None = "none",
    min_abs_percentile: float = 0.05,
    molecule_delimiter_min_size: float = math.inf,
    renumber_molecule_residues: bool = False,
    target_selector: str | None = None,
) -> ShortestPathOutputs:
    input_dir = input_dir.expanduser().resolve()
    residue_plot = resolve_output_plot_path(output)
    high_res_plot = sibling_path(residue_plot, ".high_res", ".png")
    residue_table = residue_plot.with_suffix(".tsv")
    high_res_table = sibling_path(residue_plot, ".high_res", ".tsv")

    vertices = load_vertices(input_dir / "vertices.parquet")

    edge_summary = compute_edge_analysis_summary(
        input_dir=input_dir,
        value_name=value_name,
        normalization=normalization,
        exclude_kinds=exclude_kinds,
        exclude_resnames=exclude_resnames,
        exclude_labels=exclude_labels,
        exclude_vertex_ids=exclude_vertex_ids,
        min_abs_value=min_abs_value,
        min_abs_percentile=min_abs_percentile,
        max_edges=None,
    )

    graph_data = build_scipy_graph(edge_summary.table,value_name)
    target_vertex_ids, n_target_selected, n_target_used = resolve_target_vertices(
        selector=target_selector,
        vertices=vertices,
        graph_data=graph_data,
    )
    metrics_by_vertex = compute_vertex_shortest_path_metrics_scipy(
        graph_data,
        target_vertex_ids=target_vertex_ids,
    )

    residue_rows, n_without_residue_id = collapse_metrics_to_residues(
        vertices=vertices,
        metrics_by_vertex=metrics_by_vertex,
    )
    high_res_rows = collect_high_res_rows(
        vertices=vertices,
        metrics_by_vertex=metrics_by_vertex,
    )

    residue_plot.parent.mkdir(parents=True, exist_ok=True)

    write_residue_tsv(residue_table, residue_rows, target_selector=target_selector)
    write_residue_plot(
        path=residue_plot,
        rows=residue_rows,
        value_name=edge_summary.value_name,
        normalization=edge_summary.normalization,
        molecule_delimiter_min_size=molecule_delimiter_min_size,
        renumber_molecule_residues=renumber_molecule_residues,
        target_selector=target_selector,
    )

    high_res_plot_written: Path | None = None
    high_res_table_written: Path | None = None
    if high_res_rows:
        write_high_res_tsv(high_res_table, high_res_rows, target_selector=target_selector)
        write_high_res_plot(
            path=high_res_plot,
            rows=high_res_rows,
            value_name=edge_summary.value_name,
            normalization=edge_summary.normalization,
        )
        high_res_plot_written = high_res_plot
        high_res_table_written = high_res_table

    finite_means = np.asarray(
        [m.mean_shortest_path for m in metrics_by_vertex.values() if math.isfinite(m.mean_shortest_path)],
        dtype=np.float64,
    )
    unique_count = int(len(np.unique(np.round(finite_means, decimals=12)))) if finite_means.size else 0
    n_isolated = sum(1 for metric in metrics_by_vertex.values() if metric.reachable_vertices == 0)

    return ShortestPathOutputs(
        residue_plot=residue_plot,
        high_res_plot=high_res_plot_written,
        residue_table=residue_table,
        high_res_table=high_res_table_written,
        n_vertices_used=len(graph_data.vertex_to_node),
        n_edges_used=graph_data.n_edges,
        target_selector=target_selector,
        n_target_vertices_selected=n_target_selected,
        n_target_vertices_used=n_target_used,
        n_isolated_vertices=n_isolated,
        n_vertices_without_residue_id=n_without_residue_id,
        n_high_res_vertices_plotted=len(high_res_rows),
        n_unique_vertex_means=unique_count,
        mean_min=float(np.min(finite_means)) if finite_means.size else math.nan,
        mean_median=float(np.median(finite_means)) if finite_means.size else math.nan,
        mean_max=float(np.max(finite_means)) if finite_means.size else math.nan,
    )


@dataclass(frozen=True)
class ScipyGraphData:
    matrix: object
    vertex_to_node: dict[int, int]
    node_to_vertex: dict[int, int]
    n_edges: int


def build_scipy_graph(edge_table,value_name) -> ScipyGraphData:
    try:
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise RuntimeError(
            "plot shortest_path requires scipy. Install with: pip install scipy"
        ) from exc

    vertex1 = edge_table.column("vertex1").to_numpy(zero_copy_only=False).astype(np.int64)
    vertex2 = edge_table.column("vertex2").to_numpy(zero_copy_only=False).astype(np.int64)
    values = edge_table.column("value").to_numpy(zero_copy_only=False).astype(np.float64)

    included_vertices: set[int] = set()
    edges: list[tuple[int, int, float]] = []
    zero_tolerance = np.finfo(np.float64).eps
    for v1, v2, value in zip(vertex1, vertex2, values):
        value = float(value)
        if not math.isfinite(value) or value == 0.0:
            continue
        if "dvdw" in value_name or "dcl" in value_name:
            distance = abs(value)
        else:
            distance = 1.0 / abs(value)
        if not math.isfinite(distance) or distance <= zero_tolerance:
            continue
        v1_int = int(v1)
        v2_int = int(v2)
        if v1_int == v2_int:
            continue
        edges.append((v1_int, v2_int, distance))
        included_vertices.add(v1_int)
        included_vertices.add(v2_int)

    vertex_to_node: dict[int, int] = {}
    node_to_vertex: dict[int, int] = {}
    for node_index, vertex_id in enumerate(sorted(included_vertices)):
        vertex_to_node[vertex_id] = node_index
        node_to_vertex[node_index] = vertex_id

    n = len(vertex_to_node)
    if n == 0:
        matrix = coo_matrix((0, 0), dtype=np.float64).tocsr()
        return ScipyGraphData(matrix=matrix, vertex_to_node={}, node_to_vertex={}, n_edges=0)

    row_indices: list[int] = []
    col_indices: list[int] = []
    weights: list[float] = []

    # Undirected graph: add both directions. If duplicate edges occur, scipy's
    # Dijkstra sees the stored sum after COO conversion, so we collapse duplicates
    # explicitly by keeping the minimum distance.
    best: dict[tuple[int, int], float] = {}
    for v1, v2, distance in edges:
        n1 = vertex_to_node[v1]
        n2 = vertex_to_node[v2]
        key_a = (n1, n2)
        key_b = (n2, n1)
        best[key_a] = min(distance, best.get(key_a, math.inf))
        best[key_b] = min(distance, best.get(key_b, math.inf))

    for (n1, n2), distance in best.items():
        row_indices.append(n1)
        col_indices.append(n2)
        weights.append(distance)

    matrix = coo_matrix(
        (np.asarray(weights, dtype=np.float64), (np.asarray(row_indices), np.asarray(col_indices))),
        shape=(n, n),
        dtype=np.float64,
    ).tocsr()

    return ScipyGraphData(
        matrix=matrix,
        vertex_to_node=vertex_to_node,
        node_to_vertex=node_to_vertex,
        n_edges=len(edges),
    )


def resolve_target_vertices(
    selector: str | None,
    vertices: list[VertexRecord],
    graph_data: ScipyGraphData,
) -> tuple[np.ndarray | None, int, int]:
    if selector is None or not selector.strip():
        return None, 0, 0

    columns = vertex_records_to_columns(vertices)
    try:
        mask = evaluate_vertex_selector(selector, columns, n_rows=len(vertices))
    except SelectorError as exc:
        raise RuntimeError(f"invalid --target selector: {exc}") from exc

    selected = np.asarray(
        [vertex.vertex_id for vertex, keep in zip(vertices, mask) if bool(keep)],
        dtype=np.int64,
    )
    if selected.size == 0:
        raise RuntimeError(f"--target selector matched no vertices: {selector!r}")

    used = np.asarray(
        [int(vertex_id) for vertex_id in selected if int(vertex_id) in graph_data.vertex_to_node],
        dtype=np.int64,
    )
    if used.size == 0:
        raise RuntimeError(
            "--target selector matched vertices, but none remain in the shortest-path graph "
            "after edge filtering/exclusions"
        )

    return used, int(selected.size), int(used.size)


def compute_vertex_shortest_path_metrics_scipy(
    graph_data: ScipyGraphData,
    target_vertex_ids: np.ndarray | None = None,
) -> dict[int, VertexPathMetrics]:
    try:
        from scipy.sparse.csgraph import dijkstra
    except ImportError as exc:
        raise RuntimeError(
            "plot shortest_path requires scipy. Install with: pip install scipy"
        ) from exc

    metrics: dict[int, VertexPathMetrics] = {}

    n_nodes = len(graph_data.vertex_to_node)
    if n_nodes == 0:
        return metrics

    if target_vertex_ids is None:
        distances = dijkstra(
            csgraph=graph_data.matrix,
            directed=False,
            return_predecessors=False,
            unweighted=False,
        )

        for node_index in range(n_nodes):
            vertex_id = graph_data.node_to_vertex[node_index]
            row = np.asarray(distances[node_index], dtype=np.float64)
            mask = np.isfinite(row)
            mask[node_index] = False
            values = row[mask]
            metrics[vertex_id] = path_metrics_from_values(vertex_id, values)

        return metrics

    target_nodes = np.asarray(
        [graph_data.vertex_to_node[int(vertex_id)] for vertex_id in target_vertex_ids],
        dtype=np.int64,
    )

    # The graph is undirected. Running Dijkstra from selected target nodes gives
    # distance(target -> source), which equals distance(source -> target), while
    # avoiding an all-source distance matrix when the target set is small.
    distances_from_targets = dijkstra(
        csgraph=graph_data.matrix,
        directed=False,
        indices=target_nodes,
        return_predecessors=False,
        unweighted=False,
    )
    distances_from_targets = np.asarray(distances_from_targets, dtype=np.float64)
    if distances_from_targets.ndim == 1:
        distances_from_targets = distances_from_targets[None, :]

    for source_node in range(n_nodes):
        vertex_id = graph_data.node_to_vertex[source_node]
        values = np.asarray(distances_from_targets[:, source_node], dtype=np.float64)
        finite = np.isfinite(values)

        # If the source vertex is part of the target set, do not collapse the
        # source-target metric to zero. Average to the rest of the target set.
        finite &= target_nodes != source_node

        metrics[vertex_id] = path_metrics_from_values(vertex_id, values[finite])

    return metrics


def path_metrics_from_values(vertex_id: int, values: np.ndarray) -> VertexPathMetrics:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size:
        return VertexPathMetrics(
            vertex_id=vertex_id,
            reachable_vertices=int(values.size),
            mean_shortest_path=float(np.mean(values)),
            median_shortest_path=float(np.median(values)),
            min_shortest_path=float(np.min(values)),
            max_shortest_path=float(np.max(values)),
        )

    return VertexPathMetrics(
        vertex_id=vertex_id,
        reachable_vertices=0,
        mean_shortest_path=math.nan,
        median_shortest_path=math.nan,
        min_shortest_path=math.nan,
        max_shortest_path=math.nan,
    )


@dataclass(frozen=True)
class ResidueRow:
    molecule_type: str
    molecule_instance: int | None
    residue_id: int
    residue_name: str
    n_vertices: int
    n_high_res_vertices: int
    n_residue_vertices: int
    mean_shortest_path: float
    median_shortest_path: float
    min_shortest_path: float
    max_shortest_path: float
    mean_reachable_vertices: float


@dataclass(frozen=True)
class HighResRow:
    plot_index: int
    vertex_id: int
    molecule_type: str
    molecule_instance: int | None
    residue_id: int
    residue_name: str
    atom_nr: int | None
    atom_name: str
    label: str
    reachable_vertices: int
    mean_shortest_path: float
    median_shortest_path: float
    min_shortest_path: float
    max_shortest_path: float


def collapse_metrics_to_residues(
    vertices: list[VertexRecord],
    metrics_by_vertex: dict[int, VertexPathMetrics],
) -> tuple[list[ResidueRow], int]:
    grouped: dict[tuple[str, int | None, int], list[tuple[VertexRecord, VertexPathMetrics]]] = {}
    n_without_residue_id = 0

    for vertex in vertices:
        metric = metrics_by_vertex.get(vertex.vertex_id)
        if metric is None or not math.isfinite(metric.mean_shortest_path):
            continue
        if vertex.residue_id is None:
            n_without_residue_id += 1
            continue
        grouped.setdefault((vertex.molecule_type, vertex.molecule_instance, vertex.residue_id), []).append((vertex, metric))

    rows: list[ResidueRow] = []
    for molecule_type, molecule_instance, residue_id in sorted(
        grouped,
        key=lambda key: (key[0], 10**18 if key[1] is None else key[1], key[2]),
    ):
        items = grouped[(molecule_type, molecule_instance, residue_id)]
        values = np.asarray([item[1].mean_shortest_path for item in items], dtype=np.float64)
        reachable = np.asarray([item[1].reachable_vertices for item in items], dtype=np.float64)
        residue_name = first_nonempty([item[0].residue_name for item in items])
        rows.append(
            ResidueRow(
                molecule_type=molecule_type,
                molecule_instance=molecule_instance,
                residue_id=int(residue_id),
                residue_name=residue_name,
                n_vertices=len(items),
                n_high_res_vertices=sum(1 for vertex, _ in items if vertex.kind == "atom"),
                n_residue_vertices=sum(1 for vertex, _ in items if vertex.kind == "residue"),
                mean_shortest_path=float(np.mean(values)),
                median_shortest_path=float(np.median(values)),
                min_shortest_path=float(np.min(values)),
                max_shortest_path=float(np.max(values)),
                mean_reachable_vertices=float(np.mean(reachable)),
            )
        )

    return rows, n_without_residue_id


def collect_high_res_rows(
    vertices: list[VertexRecord],
    metrics_by_vertex: dict[int, VertexPathMetrics],
) -> list[HighResRow]:
    atom_vertices = [
        vertex for vertex in vertices
        if vertex.kind == "atom" and vertex.residue_id is not None
    ]
    atom_vertices.sort(
        key=lambda vertex: (
            vertex.molecule_type,
            vertex.molecule_instance if vertex.molecule_instance is not None else 10**18,
            vertex.residue_id if vertex.residue_id is not None else 10**18,
            vertex.atom_nr if vertex.atom_nr is not None else 10**18,
            vertex.vertex_id,
        )
    )

    rows: list[HighResRow] = []
    for vertex in atom_vertices:
        metric = metrics_by_vertex.get(vertex.vertex_id)
        if metric is None or not math.isfinite(metric.mean_shortest_path):
            continue
        rows.append(
            HighResRow(
                plot_index=len(rows),
                vertex_id=vertex.vertex_id,
                molecule_type=vertex.molecule_type,
                molecule_instance=vertex.molecule_instance,
                residue_id=int(vertex.residue_id),
                residue_name=vertex.residue_name,
                atom_nr=vertex.atom_nr,
                atom_name=vertex.atom_name,
                label=vertex.label,
                reachable_vertices=metric.reachable_vertices,
                mean_shortest_path=metric.mean_shortest_path,
                median_shortest_path=metric.median_shortest_path,
                min_shortest_path=metric.min_shortest_path,
                max_shortest_path=metric.max_shortest_path,
            )
        )

    return rows


def write_residue_plot(
    path: Path,
    rows: list[ResidueRow],
    value_name: str,
    normalization: str,
    molecule_delimiter_min_size: float = math.inf,
    renumber_molecule_residues: bool = False,
    target_selector: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(
        rows,
        key=lambda row: (
            row.molecule_type,
            10**18 if row.molecule_instance is None else row.molecule_instance,
            row.residue_id,
        ),
    )
    x_plot, x_labels, jump_markers = compressed_residue_axis(
        rows,
        max_missing_residues=2,
        renumber_molecule_residues=renumber_molecule_residues,
    )
    y = np.asarray([row.mean_shortest_path for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if len(x_plot) > 0:
        plot_connected_molecule_blocks(ax, rows, x_plot, y)
        ax.scatter(x_plot, y, s=14, zorder=3)
        add_residue_axis_jump_markers(ax, jump_markers)
        add_molecule_delimiters(ax, rows, x_plot, molecule_delimiter_min_size)
        set_sparse_residue_ticks(ax, x_plot, x_labels)

    if renumber_molecule_residues:
        ax.set_xlabel("Residue index within molecule (compressed at gaps >2)")
    else:
        ax.set_xlabel("Residue ID (compressed at gaps >2)")
    ax.set_ylabel("Mean shortest-path interaction distance")
    target_suffix = "" if not target_selector else f"; target={target_selector}"
    ax.set_title(f"Shortest-path profile ({value_name}, normalization={normalization}{target_suffix})")
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compressed_residue_axis(
    rows: list[ResidueRow],
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
        if renumber_molecule_residues:
            x_labels.append(str(residue_index_in_molecule))
        else:
            x_labels.append(str(row.residue_id))
        previous_residue_id = row.residue_id
        previous_molecule_key = current_molecule_key

    return np.asarray(x_values, dtype=np.float64), x_labels, jump_markers


def molecule_block_key(row: ResidueRow) -> tuple[str, int | None]:
    return row.molecule_type, row.molecule_instance


def iter_molecule_blocks(rows: list[ResidueRow]) -> Iterable[tuple[int, int]]:
    if not rows:
        return

    start = 0
    previous = molecule_block_key(rows[0])
    for index, row in enumerate(rows[1:], start=1):
        current = molecule_block_key(row)
        if current != previous:
            yield start, index
            start = index
            previous = current
    yield start, len(rows)


def plot_connected_molecule_blocks(
    ax,
    rows: list[ResidueRow],
    x_plot: np.ndarray,
    y: np.ndarray,
) -> None:
    for start, stop in iter_molecule_blocks(rows):
        if stop - start < 2:
            continue
        ax.plot(
            x_plot[start:stop],
            y[start:stop],
            linewidth=1.0,
            alpha=0.65,
            zorder=2,
        )


def add_molecule_delimiters(
    ax,
    rows: list[ResidueRow],
    x_plot: np.ndarray,
    min_size: float,
) -> None:
    if not np.isfinite(min_size):
        return

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


def write_high_res_plot(path: Path, rows: list[HighResRow], value_name: str, normalization: str) -> None:
    import matplotlib.pyplot as plt

    x = np.asarray([row.plot_index for row in rows], dtype=np.float64)
    y = np.asarray([row.mean_shortest_path for row in rows], dtype=np.float64)

    fig_width = max(10.0, min(28.0, 0.22 * max(1, len(rows))))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.scatter(x, y, s=12)

    for boundary in high_res_residue_boundaries(rows):
        ax.axvline(boundary - 0.5, linewidth=0.6)

    ax.set_xlabel("High-resolution atom vertices ordered by residue ID and atom number")
    ax.set_ylabel("Mean shortest-path interaction distance")
    ax.set_title(f"High-resolution shortest-path detail ({value_name}, normalization={normalization})")

    tick_step = max(1, int(math.ceil(len(rows) / 80)))
    tick_indices = list(range(0, len(rows), tick_step))
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([short_atom_tick(rows[index]) for index in tick_indices], rotation=90, fontsize=6)

    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3.0)
    ax.spines["bottom"].set_linewidth(3.0)
    ax.tick_params(axis="both", width=3.0, length=7.0, direction="out")
    ax.grid(False)


def high_res_residue_boundaries(rows: list[HighResRow]) -> list[int]:
    boundaries: list[int] = []
    previous_key: tuple[int, int] | None = None
    for index, row in enumerate(rows):
        key = (row.residue_id, row.atom_nr if row.atom_nr is not None else -1)
        residue_key = (row.residue_id, 0)
        if previous_key is not None and residue_key != previous_key:
            boundaries.append(index)
        previous_key = residue_key
    return boundaries


def short_atom_tick(row: HighResRow) -> str:
    if row.atom_name:
        return f"{row.residue_id}:{row.atom_name}"
    if row.atom_nr is not None:
        return f"{row.residue_id}:{row.atom_nr}"
    return f"{row.residue_id}:{row.vertex_id}"


def write_residue_tsv(path: Path, rows: list[ResidueRow], target_selector: str | None = None) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "molecule_type",
            "molecule_instance",
            "residue_id",
            "residue_name",
            "target_selector",
            "n_vertices",
            "n_high_res_vertices",
            "n_residue_vertices",
            "mean_shortest_path",
            "median_shortest_path",
            "min_shortest_path",
            "max_shortest_path",
            "mean_reachable_vertices",
        ])
        for row in rows:
            writer.writerow([
                row.molecule_type,
                "" if row.molecule_instance is None else row.molecule_instance,
                row.residue_id,
                row.residue_name,
                "" if target_selector is None else target_selector,
                row.n_vertices,
                row.n_high_res_vertices,
                row.n_residue_vertices,
                format_float(row.mean_shortest_path),
                format_float(row.median_shortest_path),
                format_float(row.min_shortest_path),
                format_float(row.max_shortest_path),
                format_float(row.mean_reachable_vertices),
            ])


def write_high_res_tsv(path: Path, rows: list[HighResRow], target_selector: str | None = None) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "plot_index",
            "vertex_id",
            "molecule_type",
            "molecule_instance",
            "residue_id",
            "residue_name",
            "atom_nr",
            "atom_name",
            "label",
            "target_selector",
            "reachable_vertices",
            "mean_shortest_path",
            "median_shortest_path",
            "min_shortest_path",
            "max_shortest_path",
        ])
        for row in rows:
            writer.writerow([
                row.plot_index,
                row.vertex_id,
                row.molecule_type,
                "" if row.molecule_instance is None else row.molecule_instance,
                row.residue_id,
                row.residue_name,
                "" if row.atom_nr is None else row.atom_nr,
                row.atom_name,
                row.label,
                "" if target_selector is None else target_selector,
                row.reachable_vertices,
                format_float(row.mean_shortest_path),
                format_float(row.median_shortest_path),
                format_float(row.min_shortest_path),
                format_float(row.max_shortest_path),
            ])


def load_vertices(path: Path) -> list[VertexRecord]:
    table = pq.read_table(path)
    columns = table.column_names
    data = {name: table.column(name).to_pylist() for name in columns}

    n_rows = table.num_rows
    records: list[VertexRecord] = []
    for row in range(n_rows):
        vertex_id = to_int(get_value(data, "vertex_id", row, row), default=row)
        records.append(
            VertexRecord(
                vertex_id=vertex_id,
                kind=to_str(get_value(data, "kind", row, "")),
                label=to_str(get_value(data, "label", row, "")),
                residue_name=to_str(get_value(data, "residue_name", row, "")),
                residue_id=to_optional_int(get_value(data, "residue_id", row, None)),
                molecule_type=to_str(get_value(data, "molecule_type", row, "")),
                molecule_instance=to_optional_int(get_value(data, "molecule_instance", row, None)),
                atom_nr=to_optional_int(get_value(data, "atom_nr", row, None)),
                atom_name=to_str(get_value(data, "atom_name", row, "")),
            )
        )

    records.sort(key=lambda record: record.vertex_id)
    return records


def get_value(data: dict[str, list], name: str, row: int, default):
    if name not in data:
        return default
    return data[name][row]


def to_str(value) -> str:
    if value is None:
        return ""
    return str(value)


def to_optional_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def to_int(value, default: int) -> int:
    optional = to_optional_int(value)
    if optional is None:
        return default
    return optional


def first_nonempty(values: Iterable[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.12g}"


def resolve_output_plot_path(output: Path) -> Path:
    output = output.expanduser()
    if output.suffix.lower() in {".png", ".pdf", ".svg"}:
        return output
    return output / "shortest_path.png"


def sibling_path(path: Path, insert: str, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{insert}{suffix}")
