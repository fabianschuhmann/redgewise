from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq

from redgewise.analysis_values import (
    VALUE_COLUMNS,
    canonical_normalization,
    canonical_value_name,
    compute_vertex_member_counts,
)
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
class RegionSpec:
    index: int
    selector: str
    label: str
    vertex_mask: np.ndarray
    residue_keys: tuple[tuple[str, int | None, int, str], ...]
    residue_index_by_vertex: dict[int, int]


@dataclass(frozen=True)
class PairMatrix:
    source_region: RegionSpec
    target_region: RegionSpec
    frames: np.ndarray
    matrix: np.ndarray
    edge_counts: np.ndarray


@dataclass(frozen=True)
class RavePlotOutputs:
    plot: Path
    table: Path
    n_regions: int
    n_region_pairs: int
    n_frames: int
    n_edges_considered: int
    value_name: str
    normalization: str

OKABE_ITO = {
    "blue": (0.0 / 255.0, 114.0 / 255.0, 178.0 / 255.0),
    "orange": (230.0 / 255.0, 159.0 / 255.0, 0.0 / 255.0),
    "green": (0.0 / 255.0, 158.0 / 255.0, 115.0 / 255.0),
    "purple": (204.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0),
}


def run_plot_rave(args) -> None:
    outputs = plot_rave(
        input_dir=args.input,
        output=args.output,
        value_name=args.value,
        normalization=args.normalize,
        region_selectors=args.region,
        region_labels=args.region_label,
        alpha=args.alpha,
        darkmode=args.darkmode,
    )

    print("RAVE plot written:")
    print(f"  plot:              {outputs.plot}")
    print(f"  table:             {outputs.table}")
    print(f"  regions:           {outputs.n_regions}")
    print(f"  region pairs:      {outputs.n_region_pairs}")
    print(f"  frames:            {outputs.n_frames}")
    print(f"  edges considered:  {outputs.n_edges_considered}")
    print(f"  value:             {outputs.value_name}")
    print(f"  normalization:     {outputs.normalization}")


def plot_rave(
    input_dir: Path,
    output: Path,
    value_name: str = "vdw+cl",
    normalization: str = "none",
    region_selectors: Iterable[str] = (),
    region_labels: Iterable[str] | None = None,
    alpha: float = 0.9,
    darkmode: bool = False,
) -> RavePlotOutputs:
    input_dir = input_dir.expanduser().resolve()
    plot_path = resolve_output_plot_path(output)
    table_path = plot_path.with_suffix(".tsv")

    value_name = canonical_value_name(value_name)
    normalization = canonical_normalization(normalization)
    alpha = validate_alpha(alpha)

    selectors = [selector for selector in region_selectors if str(selector).strip()]
    if len(selectors) < 2:
        raise ValueError("plot rave requires at least two --region selectors")

    labels = list(region_labels or [])
    if labels and len(labels) != len(selectors):
        raise ValueError("--region-label must be repeated exactly as often as --region, or not used")

    vertices = read_vertices(input_dir / "vertices.parquet")
    vertex_table = pq.read_table(input_dir / "vertices.parquet")
    vertex_members = pq.read_table(input_dir / "vertex_members.parquet")
    edges_table = pq.read_table(input_dir / "edges.parquet")

    regions = resolve_regions(vertices, selectors, labels)
    edge_lookup = read_edges(edges_table)
    normalization_factor_by_edge = compute_edge_normalization_factors(
        edges_table=edges_table,
        vertices=vertices,
        vertex_table=vertex_table,
        vertex_members=vertex_members,
        normalization=normalization,
    )

    frames = read_frame_index(input_dir=input_dir)
    frame_to_row = {int(frame): index for index, frame in enumerate(frames)}

    pair_matrices, n_edges_considered = compute_pair_matrices(
        input_dir=input_dir,
        value_name=value_name,
        normalization=normalization,
        edge_lookup=edge_lookup,
        normalization_factor_by_edge=normalization_factor_by_edge,
        regions=regions,
        frames=frames,
        frame_to_row=frame_to_row,
    )

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    write_rave_table(
        path=table_path,
        pair_matrices=pair_matrices,
        value_name=value_name,
        normalization=normalization,
    )
    write_rave_plot(
        path=plot_path,
        pair_matrices=pair_matrices,
        value_name=value_name,
        normalization=normalization,
        alpha=alpha,
        darkmode=darkmode,
    )

    return RavePlotOutputs(
        plot=plot_path,
        table=table_path,
        n_regions=len(regions),
        n_region_pairs=len(pair_matrices) // 2,
        n_frames=len(frames),
        n_edges_considered=n_edges_considered,
        value_name=value_name,
        normalization=normalization,
    )


def resolve_output_plot_path(output: Path) -> Path:
    output = output.expanduser()
    if output.suffix:
        return output.resolve()
    return (output / "rave.png").resolve()


def validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError("--alpha must be > 0 and <= 1")
    return value


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


def resolve_regions(
    vertices: list[VertexRecord],
    selectors: list[str],
    labels: list[str],
) -> list[RegionSpec]:
    columns = vertex_records_to_columns(vertices)
    n_vertices = len(vertices)
    assigned = np.full(n_vertices, -1, dtype=np.int32)
    regions: list[RegionSpec] = []

    for index, selector in enumerate(selectors):
        try:
            mask = evaluate_vertex_selector(selector, columns, n_rows=n_vertices)
        except SelectorError as exc:
            raise ValueError(f"invalid --region selector {selector!r}: {exc}") from exc
        mask = np.asarray(mask, dtype=bool)
        n_selected = int(np.count_nonzero(mask))
        if n_selected == 0:
            raise ValueError(f"--region selector matched no vertices: {selector!r}")

        overlap = np.where(mask & (assigned >= 0))[0]
        if len(overlap) > 0:
            first_vertex = vertices[int(overlap[0])]
            previous_region = int(assigned[int(overlap[0])])
            raise ValueError(
                "--region selectors must be disjoint; "
                f"region {index + 1} {selector!r} overlaps region {previous_region + 1} "
                f"at vertex_id={first_vertex.vertex_id} label={first_vertex.label!r}"
            )
        assigned[mask] = index

        residue_keys, residue_index_by_vertex = region_residue_index(vertices, mask)
        if not residue_keys:
            raise ValueError(
                f"--region selector matched vertices, but none have residue_id: {selector!r}"
            )
        label = labels[index] if labels else f"R{index + 1}"
        regions.append(
            RegionSpec(
                index=index,
                selector=selector,
                label=label,
                vertex_mask=mask,
                residue_keys=tuple(residue_keys),
                residue_index_by_vertex=residue_index_by_vertex,
            )
        )
    return regions


def region_residue_index(
    vertices: list[VertexRecord],
    mask: np.ndarray,
) -> tuple[list[tuple[str, int | None, int, str]], dict[int, int]]:
    key_by_vertex: dict[int, tuple[str, int | None, int, str]] = {}
    for vertex in vertices:
        if not mask[vertex.vertex_id]:
            continue
        if vertex.residue_id is None:
            continue
        key = (
            vertex.molecule_type,
            vertex.molecule_instance,
            int(vertex.residue_id),
            vertex.residue_name,
        )
        key_by_vertex[vertex.vertex_id] = key

    residue_keys = sorted(
        set(key_by_vertex.values()),
        key=lambda key: (key[0], 10**18 if key[1] is None else key[1], key[2], key[3]),
    )
    index_by_key = {key: index for index, key in enumerate(residue_keys)}
    residue_index_by_vertex = {
        vertex_id: index_by_key[key]
        for vertex_id, key in key_by_vertex.items()
    }
    return residue_keys, residue_index_by_vertex


def read_edges(edges_table) -> dict[int, tuple[int, int]]:
    edge_key = edges_table.column("edge_key").to_numpy(zero_copy_only=False).astype(np.int64)
    vertex1 = edges_table.column("vertex1").to_numpy(zero_copy_only=False).astype(np.int64)
    vertex2 = edges_table.column("vertex2").to_numpy(zero_copy_only=False).astype(np.int64)
    return {int(key): (int(v1), int(v2)) for key, v1, v2 in zip(edge_key, vertex1, vertex2)}


def compute_edge_normalization_factors(
    edges_table,
    vertices: list[VertexRecord],
    vertex_table,
    vertex_members,
    normalization: str,
) -> dict[int, float]:
    edge_key = edges_table.column("edge_key").to_numpy(zero_copy_only=False).astype(np.int64)
    vertex1 = edges_table.column("vertex1").to_numpy(zero_copy_only=False).astype(np.int64)
    vertex2 = edges_table.column("vertex2").to_numpy(zero_copy_only=False).astype(np.int64)

    if normalization == "none" or normalization == "per_atom_pair":
        return {int(key): 1.0 for key in edge_key}

    member_counts = compute_vertex_member_counts(
        n_vertices=vertex_table.num_rows,
        vertex_members=vertex_members,
    )

    factors: dict[int, float] = {}
    for key, v1, v2 in zip(edge_key, vertex1, vertex2):
        n1 = float(member_counts[int(v1)])
        n2 = float(member_counts[int(v2)])
        if normalization == "per_vertex_member_sqrt":
            denominator = math.sqrt(n1 * n2)
        elif normalization == "per_vertex_member_product":
            denominator = n1 * n2
        elif normalization in {"per_coarse_member_sqrt", "per_coarse_member_product"}:
            f1 = n1 if vertices[int(v1)].kind in {"low_res", "bundle"} else 1.0
            f2 = n2 if vertices[int(v2)].kind in {"low_res", "bundle"} else 1.0
            denominator = f1 * f2
            if normalization == "per_coarse_member_sqrt":
                denominator = math.sqrt(denominator)
        else:  # pragma: no cover - guarded by canonical_normalization
            raise ValueError(f"unknown normalization: {normalization}")
        factors[int(key)] = denominator if denominator > 0.0 else 1.0
    return factors


def read_frame_index(input_dir: Path) -> np.ndarray:
    frames: set[int] = set()
    for path in sorted((input_dir / "values").glob("*.parquet")):
        table = pq.read_table(path, columns=["frame"])
        frames.update(int(frame) for frame in np.unique(table.column("frame").to_numpy(zero_copy_only=False)))
    if not frames:
        raise ValueError(f"no frames found in {input_dir / 'values'}")
    return np.asarray(sorted(frames), dtype=np.int64)


def compute_pair_matrices(
    input_dir: Path,
    value_name: str,
    normalization: str,
    edge_lookup: dict[int, tuple[int, int]],
    normalization_factor_by_edge: dict[int, float],
    regions: list[RegionSpec],
    frames: np.ndarray,
    frame_to_row: dict[int, int],
) -> tuple[list[PairMatrix], int]:
    n_vertices = max(max(v1, v2) for v1, v2 in edge_lookup.values()) + 1 if edge_lookup else 0
    region_by_vertex = np.full(n_vertices, -1, dtype=np.int32)
    for region in regions:
        limit = min(n_vertices, len(region.vertex_mask))
        region_by_vertex[:limit][region.vertex_mask[:limit]] = region.index

    matrices_by_direction: dict[tuple[int, int], np.ndarray] = {}
    counts_by_direction: dict[tuple[int, int], np.ndarray] = {}
    region_by_index = {region.index: region for region in regions}

    for source, target in ((a.index, b.index) for a, b in permutations_regions(regions)):
        source_region = region_by_index[source]
        shape = (len(frames), len(source_region.residue_keys))
        matrices_by_direction[(source, target)] = np.zeros(shape, dtype=np.float64)
        counts_by_direction[(source, target)] = np.zeros(shape, dtype=np.int32)

    columns = VALUE_COLUMNS[value_name]
    n_edges_considered = 0

    for path in sorted((input_dir / "values").glob("*.parquet")):
        table = pq.read_table(path, columns=["frame", "edge_key", *columns, "n_atom_pairs"])
        frame_array = table.column("frame").to_numpy(zero_copy_only=False).astype(np.int64)
        edge_key_array = table.column("edge_key").to_numpy(zero_copy_only=False).astype(np.int64)
        n_atom_pairs_array = table.column("n_atom_pairs").to_numpy(zero_copy_only=False).astype(np.float64)
        value_array = np.zeros(len(edge_key_array), dtype=np.float64)
        for column in columns:
            value_array += table.column(column).to_numpy(zero_copy_only=False).astype(np.float64)

        for frame, edge_key, raw_value, n_atom_pairs in zip(
            frame_array,
            edge_key_array,
            value_array,
            n_atom_pairs_array,
        ):
            pair = edge_lookup.get(int(edge_key))
            if pair is None:
                continue
            v1, v2 = pair
            if v1 >= len(region_by_vertex) or v2 >= len(region_by_vertex):
                continue
            r1 = int(region_by_vertex[v1])
            r2 = int(region_by_vertex[v2])
            if r1 < 0 or r2 < 0 or r1 == r2:
                continue

            value = float(raw_value)
            if normalization == "per_atom_pair":
                if n_atom_pairs <= 0.0:
                    continue
                value /= float(n_atom_pairs)
            else:
                value /= normalization_factor_by_edge.get(int(edge_key), 1.0)
            if not math.isfinite(value):
                continue

            row = frame_to_row[int(frame)]
            add_directional_value(
                matrices_by_direction=matrices_by_direction,
                counts_by_direction=counts_by_direction,
                region_by_index=region_by_index,
                source_vertex=v1,
                source_region_index=r1,
                target_region_index=r2,
                frame_row=row,
                value=value,
            )
            add_directional_value(
                matrices_by_direction=matrices_by_direction,
                counts_by_direction=counts_by_direction,
                region_by_index=region_by_index,
                source_vertex=v2,
                source_region_index=r2,
                target_region_index=r1,
                frame_row=row,
                value=value,
            )
            n_edges_considered += 1

    pair_matrices: list[PairMatrix] = []
    for region_a, region_b in combinations(regions, 2):
        pair_matrices.append(
            PairMatrix(
                source_region=region_a,
                target_region=region_b,
                frames=frames,
                matrix=matrices_by_direction[(region_a.index, region_b.index)],
                edge_counts=counts_by_direction[(region_a.index, region_b.index)],
            )
        )
        pair_matrices.append(
            PairMatrix(
                source_region=region_b,
                target_region=region_a,
                frames=frames,
                matrix=matrices_by_direction[(region_b.index, region_a.index)],
                edge_counts=counts_by_direction[(region_b.index, region_a.index)],
            )
        )
    return pair_matrices, n_edges_considered


def permutations_regions(regions: list[RegionSpec]):
    for source in regions:
        for target in regions:
            if source.index != target.index:
                yield source, target


def add_directional_value(
    matrices_by_direction: dict[tuple[int, int], np.ndarray],
    counts_by_direction: dict[tuple[int, int], np.ndarray],
    region_by_index: dict[int, RegionSpec],
    source_vertex: int,
    source_region_index: int,
    target_region_index: int,
    frame_row: int,
    value: float,
) -> None:
    source_region = region_by_index[source_region_index]
    residue_index = source_region.residue_index_by_vertex.get(source_vertex)
    if residue_index is None:
        return
    key = (source_region_index, target_region_index)
    matrices_by_direction[key][frame_row, residue_index] += value
    counts_by_direction[key][frame_row, residue_index] += 1


def write_rave_table(
    path: Path,
    pair_matrices: list[PairMatrix],
    value_name: str,
    normalization: str,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "frame",
                "source_region",
                "target_region",
                "source_selector",
                "target_selector",
                "molecule_type",
                "molecule_instance",
                "residue_id",
                "residue_name",
                "residue_index",
                "value",
                "n_edges",
                "value_name",
                "normalization",
            ]
        )
        for pair in pair_matrices:
            for residue_index, residue_key in enumerate(pair.source_region.residue_keys):
                molecule_type, molecule_instance, residue_id, residue_name = residue_key
                values = pair.matrix[:, residue_index]
                counts = pair.edge_counts[:, residue_index]
                for frame_index, frame in enumerate(pair.frames):
                    value = float(values[frame_index])
                    n_edges = int(counts[frame_index])
                    if value == 0.0 and n_edges == 0:
                        continue
                    writer.writerow(
                        [
                            int(frame),
                            pair.source_region.label,
                            pair.target_region.label,
                            pair.source_region.selector,
                            pair.target_region.selector,
                            molecule_type,
                            "" if molecule_instance is None else molecule_instance,
                            residue_id,
                            residue_name,
                            residue_index + 1,
                            format_float(value),
                            n_edges,
                            value_name,
                            normalization,
                        ]
                    )



def write_rave_plot(
    path: Path,
    pair_matrices: list[PairMatrix],
    value_name: str,
    normalization: str,
    alpha: float,
    darkmode: bool,
) -> None:
    import matplotlib.pyplot as plt

    if not pair_matrices:
        raise ValueError("no region-pair matrices to plot")

    pair_groups: list[tuple[PairMatrix, PairMatrix]] = []
    for i in range(0, len(pair_matrices), 2):
        pair_groups.append((pair_matrices[i], pair_matrices[i + 1]))

    vmax = global_signed_vmax(pair_matrices)
    forward_cmap, reverse_cmap, norm = make_rave_colormaps_and_norm(vmax, darkmode=darkmode)

    n_panels = len(pair_groups)
    ncols = min(3, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    fig_width = max(6.4, 5.6 * ncols + 1.8)
    fig_height = max(4.0, 4.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)

    flat_axes = list(axes.ravel())
    for axis in flat_axes[n_panels:]:
        axis.set_visible(False)

    used_axes = []
    for axis, pair_index, (forward, reverse) in zip(flat_axes, range(1, n_panels + 1), pair_groups):
        draw_rave_pair_panel(
            axis=axis,
            forward=forward,
            reverse=reverse,
            forward_cmap=forward_cmap,
            reverse_cmap=reverse_cmap,
            norm=norm,
            alpha=alpha,
            darkmode=darkmode,
            title=f"{pair_index:02d}: {forward.source_region.label}↔{forward.target_region.label}",
            show_ylabel=True,
        )
        used_axes.append(axis)

    fig.suptitle(
        f"RAVE direct-neighbor interaction heatmap ({value_name})",
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 0.94))
    add_direction_colorbars(
        figure=fig,
        norm=norm,
        forward_cmap=forward_cmap,
        reverse_cmap=reverse_cmap,
        label_forward="first direction",
        label_reverse="reverse direction",
        left=0.860,
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    write_single_rave_pair_plots(
        output_path=path,
        pair_groups=pair_groups,
        forward_cmap=forward_cmap,
        reverse_cmap=reverse_cmap,
        norm=norm,
        alpha=alpha,
        darkmode=darkmode,
        value_name=value_name,
        normalization=normalization,
        vmax=vmax,
    )


def global_signed_vmax(pair_matrices: list[PairMatrix]) -> float:
    vmax = 0.0
    for pair in pair_matrices:
        if pair.matrix.size == 0:
            continue
        local = float(np.nanmax(np.abs(pair.matrix)))
        if math.isfinite(local):
            vmax = max(vmax, local)
    return vmax if vmax > 0.0 else 1.0


def _blend_with_white(color: tuple[float, float, float], fraction: float) -> tuple[float, float, float]:
    return tuple((1.0 - fraction) * channel + fraction * 1.0 for channel in color)



class SignedRootNorm:
    """Factory returning a matplotlib Normalize subclass instance.

    RAVE uses a fixed signed root transform with exponent 0.4. This is slightly
    stronger than a square-root transform and makes weaker signed interactions
    more visible without changing the reported data units.

        -vmax -> 0.0
         0    -> 0.5
        +vmax -> 1.0

    The visual coordinate is sign(x) * (abs(x) / vmax)**0.4, while inverse()
    maps colorbar positions back into original data units.
    """

    exponent = 0.4

    def __new__(cls, vmax: float):
        from matplotlib.colors import Normalize

        class _SignedRootNormalize(Normalize):
            def __init__(self, vmax_value: float):
                vmax_float = float(vmax_value)
                if not math.isfinite(vmax_float) or vmax_float <= 0.0:
                    vmax_float = 1.0
                self.vmax_abs = vmax_float
                self.exponent = SignedRootNorm.exponent
                super().__init__(vmin=-vmax_float, vmax=vmax_float, clip=False)

            def __call__(self, value, clip=None):
                values = np.ma.asarray(value, dtype=np.float64)
                signed = (
                    np.ma.array(np.sign(values), mask=np.ma.getmaskarray(values))
                    * ((np.ma.abs(values) / self.vmax_abs) ** self.exponent)
                )
                mapped = 0.5 + 0.5 * signed
                return np.ma.masked_invalid(mapped)

            def inverse(self, value):
                values = np.asarray(value, dtype=np.float64)
                signed = 2.0 * values - 1.0
                return np.sign(signed) * (np.abs(signed) ** (1.0 / self.exponent)) * self.vmax_abs

        return _SignedRootNormalize(vmax)



def make_rave_colormaps_and_norm(vmax: float, darkmode: bool = False):
    from matplotlib.colors import LinearSegmentedColormap

    if darkmode:
        # Same three-stop structure as light mode, but with black at zero.
        # The signed square-root norm handles contrast near zero; the colormap
        # itself stays simple and comparable.
        center = (0.0, 0.0, 0.0)
        forward_negative = _blend_with_white(OKABE_ITO["blue"], 0.35)
        forward_positive = _blend_with_white(OKABE_ITO["orange"], 0.25)
        reverse_negative = _blend_with_white(OKABE_ITO["green"], 0.30)
        reverse_positive = _blend_with_white(OKABE_ITO["purple"], 0.35)
    else:
        center = (1.0, 1.0, 1.0)
        forward_negative = OKABE_ITO["blue"]
        forward_positive = OKABE_ITO["orange"]
        reverse_negative = OKABE_ITO["green"]
        reverse_positive = OKABE_ITO["purple"]

    forward_cmap = LinearSegmentedColormap.from_list(
        "redgewise_rave_forward",
        [(0.0, forward_negative), (0.5, center), (1.0, forward_positive)],
        N=256,
    )
    reverse_cmap = LinearSegmentedColormap.from_list(
        "redgewise_rave_reverse",
        [(0.0, reverse_negative), (0.5, center), (1.0, reverse_positive)],
        N=256,
    )
    norm = SignedRootNorm(vmax=float(vmax))
    return forward_cmap, reverse_cmap, norm


def draw_rave_pair_panel(
    axis,
    forward: PairMatrix,
    reverse: PairMatrix,
    forward_cmap,
    reverse_cmap,
    norm,
    alpha: float,
    darkmode: bool,
    title: str,
    show_ylabel: bool,
) -> None:
    if darkmode:
        axis.set_facecolor("black")

    draw_signed_matrix_overlay(
        axis=axis,
        matrix=forward.matrix,
        frames=forward.frames,
        cmap=forward_cmap,
        norm=norm,
        alpha=alpha,
        zorder=1,
    )
    draw_signed_matrix_overlay(
        axis=axis,
        matrix=reverse.matrix,
        frames=reverse.frames,
        cmap=reverse_cmap,
        norm=norm,
        alpha=alpha,
        zorder=2,
    )

    x_max = max(forward.matrix.shape[1], reverse.matrix.shape[1])
    y_max = max(1, len(forward.frames))
    axis.set_xlim(-0.5, float(x_max) - 0.5)
    axis.set_ylim(-0.5, float(y_max) - 0.5)
    axis.set_title(title)
    axis.set_xlabel("Residue index in source")
    if show_ylabel:
        axis.set_ylabel("Frame")
    set_sparse_ticks(axis, x_max, forward.frames)
    clean_axes(axis)



def draw_signed_matrix_overlay(
    axis,
    matrix: np.ndarray,
    frames: np.ndarray,
    cmap,
    norm,
    alpha: float,
    zorder: int,
) -> None:
    if matrix.size == 0:
        return

    values = np.asarray(matrix, dtype=np.float64)
    alpha_map = np.full(values.shape, float(alpha), dtype=np.float64)
    alpha_map[~np.isfinite(values)] = 0.0

    # Exact zeros are transparent. The color scale itself is still white at zero;
    # transparency prevents one all-zero directional matrix from washing out the other overlay.
    alpha_map[values == 0.0] = 0.0

    masked = np.ma.masked_invalid(values)
    axis.imshow(
        masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(-0.5, values.shape[1] - 0.5, -0.5, len(frames) - 0.5),
        cmap=cmap,
        norm=norm,
        alpha=alpha_map,
        zorder=zorder,
    )


def format_colorbar_tick(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    if abs(value) < 1.0e-12:
        return "0"

    abs_value = abs(value)
    if abs_value >= 1000.0:
        return f"{value:.0f}"
    if abs_value >= 100.0:
        return f"{value:.0f}"
    if abs_value >= 10.0:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if abs_value >= 1.0:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if abs_value >= 0.01:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.1e}"


def add_direction_colorbars(
    figure,
    norm,
    forward_cmap,
    reverse_cmap,
    label_forward: str,
    label_reverse: str,
    left: float,
) -> None:
    from matplotlib.cm import ScalarMappable

    height = 0.58
    bottom = 0.20
    width = 0.013
    gap = 0.0

    cax_forward = figure.add_axes([left, bottom, width, height])
    cax_reverse = figure.add_axes([left + width + gap, bottom, width, height])

    sm_forward = ScalarMappable(norm=norm, cmap=forward_cmap)
    sm_forward.set_array([])
    sm_reverse = ScalarMappable(norm=norm, cmap=reverse_cmap)
    sm_reverse.set_array([])

    cbar_forward = figure.colorbar(sm_forward, cax=cax_forward)
    cbar_reverse = figure.colorbar(sm_reverse, cax=cax_reverse)

    vmax = float(getattr(norm, "vmax_abs", max(abs(norm.vmin), abs(norm.vmax))))
    exponent = float(getattr(norm, "exponent", 0.5))

    # Data-value ticks placed at visually even positions under the root transform.
    half_tick_value = vmax * (0.5 ** (1.0 / exponent))
    ticks = [-vmax, -half_tick_value, 0.0, half_tick_value, vmax]

    # The ticked scale is on the outside/right colorbar so labels cannot overlap
    # the rightmost subplot. The left colorbar is the unticked companion scale.
    cbar_forward.set_ticks([])
    cbar_forward.set_label("")
    cbar_forward.ax.tick_params(left=False, right=False, labelleft=False, labelright=False)

    cbar_reverse.set_ticks(ticks)
    cbar_reverse.set_ticklabels([format_colorbar_tick(tick) for tick in ticks])
    cbar_reverse.set_label("One way interaction value")
    cbar_reverse.ax.yaxis.set_ticks_position("right")
    cbar_reverse.ax.yaxis.set_label_position("right")
    cbar_reverse.ax.tick_params(left=False, right=True, labelleft=False, labelright=True)


def write_single_rave_pair_plots(
    output_path: Path,
    pair_groups: list[tuple[PairMatrix, PairMatrix]],
    forward_cmap,
    reverse_cmap,
    norm,
    alpha: float,
    darkmode: bool,
    value_name: str,
    normalization: str,
    vmax: float,
) -> None:
    import matplotlib.pyplot as plt

    for pair_index, (forward, reverse) in enumerate(pair_groups, start=1):
        single_path = output_path.with_name(
            f"{output_path.stem}_{pair_index:02d}{output_path.suffix or '.png'}"
        )
        fig, axis = plt.subplots(1, 1, figsize=(6.8, 4.6))
        draw_rave_pair_panel(
            axis=axis,
            forward=forward,
            reverse=reverse,
            forward_cmap=forward_cmap,
            reverse_cmap=reverse_cmap,
            norm=norm,
            alpha=alpha,
            darkmode=darkmode,
            title=f"{forward.source_region.label}↔{forward.target_region.label}",
            show_ylabel=True,
        )
        fig.suptitle(
            f"RAVE {value_name}, normalization={normalization}, |scale|={vmax:.4g}",
            y=0.98,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.80, 0.93))
        add_direction_colorbars(
            figure=fig,
            norm=norm,
            forward_cmap=forward_cmap,
            reverse_cmap=reverse_cmap,
            label_forward=f"{forward.source_region.label}→{forward.target_region.label}",
            label_reverse=f"{reverse.source_region.label}→{reverse.target_region.label}",
            left=0.835,
        )
        fig.savefig(single_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def set_sparse_ticks(axis, n_x: int, frames: np.ndarray) -> None:
    if n_x > 0:
        x_step = max(1, int(math.ceil(n_x / 12)))
        x_ticks = list(range(0, n_x, x_step))
        if n_x - 1 not in x_ticks:
            x_ticks.append(n_x - 1)
        axis.set_xticks(x_ticks)
        axis.set_xticklabels([str(index + 1) for index in x_ticks], rotation=90, fontsize=7)
    if len(frames) > 0:
        y_step = max(1, int(math.ceil(len(frames) / 8)))
        y_ticks = list(range(0, len(frames), y_step))
        if len(frames) - 1 not in y_ticks:
            y_ticks.append(len(frames) - 1)
        axis.set_yticks(y_ticks)
        axis.set_yticklabels([str(int(frames[index])) for index in y_ticks], fontsize=7)


def clean_axes(axis) -> None:
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(2.0)
    axis.spines["bottom"].set_linewidth(2.0)
    axis.tick_params(axis="both", width=2.0, length=5.0)


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.15g}"
