from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from redgewise import __version__
from redgewise.selectors import selector_help_text


KIND_PRECEDENCE = {
    "atom": 0,
    "low_res": 1,
    "bundle": 2,
    "residue": 3,
}

KIND_DISPLAY = {
    "atom": "high_res/atom",
    "low_res": "low_res",
    "bundle": "bundle",
    "residue": "default/residue",
}

SELECTOR_INVENTORY_FIELDS = [
    ("kind", "kind", "count"),
    ("molecule_type", "molecule_type / moltype", "count"),
    ("molecule_instance", "molecule_instance / molinstance", "value"),
    ("residue_name", "residue_name / resname", "count"),
    ("atom_name", "atom_name / name", "count"),
    ("atom_type", "atom_type / type", "count"),
]


def run_info(args: argparse.Namespace) -> int:
    input_dir = getattr(args, "input", None)
    requested_list = bool(getattr(args, "list", False))
    requested_selector = bool(getattr(args, "selector", False))
    long_output = bool(getattr(args, "long", False))

    if input_dir is None:
        print(f"redgewise {__version__}")
        print("REdgEwise: Residue Energy edge-wise analysis")
        print()
        if requested_selector:
            print(selector_help_text().rstrip())
        else:
            print("Usage:")
            print("  redgewise info -i OUTPUT --list")
            print("  redgewise info -i OUTPUT --selector")
            print("  redgewise info -i OUTPUT --list --selector")
            print("  redgewise info --selector")
        return 0

    input_dir = input_dir.expanduser().resolve()
    validate_output_dir(input_dir)

    if not requested_list and not requested_selector:
        requested_list = True

    metadata = read_metadata(input_dir)
    vertices = pq.read_table(input_dir / "vertices.parquet").to_pydict()
    vertex_member_counts = read_vertex_member_counts(input_dir)

    if requested_list:
        print_output_listing(
            input_dir=input_dir,
            metadata=metadata,
            vertices=vertices,
            vertex_member_counts=vertex_member_counts,
            long_output=long_output,
        )

    if requested_selector:
        if requested_list:
            print()
        print_selector_inventory(vertices=vertices, max_values=None if long_output else 25)

    return 0


def print_output_listing(
    input_dir: Path,
    metadata: dict[str, Any],
    vertices: dict[str, list[Any]],
    vertex_member_counts: dict[int, int],
    long_output: bool,
) -> None:
    print("redgewise info")
    print()
    print("Input:")
    print(f"  {input_dir}")
    print()

    print("Network:")
    print(f"  atoms:       {metadata_value(metadata, 'n_atoms', sum(vertex_member_counts.values()))}")
    print(f"  vertices:    {len(vertices['vertex_id'])}")
    print(f"  edges:       {metadata_value(metadata, 'n_edges', count_edges(input_dir))}")
    print(f"  frames:      {count_frames(input_dir)}")
    print(f"  value rows:  {metadata_value(metadata, 'n_value_rows', count_value_rows(input_dir))}")
    print()

    print("Resolution precedence:")
    print("  high_res/atom > low_res > bundle > default/residue")
    print()

    print("Resolution layers:")
    kind_counts = Counter(str(kind) for kind in vertices["kind"])
    for kind in sorted(kind_counts, key=kind_sort_key):
        print(f"  {KIND_DISPLAY.get(kind, kind):16s} {kind_counts[kind]:8d}")
    print()

    print("Vertices:")
    print(format_vertex_header())

    rows = iter_sorted_vertex_rows(vertices, vertex_member_counts)
    displayed_rows = rows if long_output else truncate_vertex_rows(rows)

    for row in displayed_rows:
        print(format_vertex_row(row))

    if not long_output and len(displayed_rows) < len(rows):
        remaining = len(rows) - len(displayed_rows)
        print(f"  ... {remaining} more vertices; use --long to show all")


def print_selector_inventory(vertices: dict[str, list[Any]], max_values: int | None = 25) -> None:
    print("Selector inventory")
    print()
    print("Use these values with selector-aware commands, for example:")
    print("  redgewise plot shortest_path -i OUTPUT -o shortest_path.png --target \"molecule_instance 0\"")
    print()

    n_vertices = len(vertices.get("vertex_id", []))
    print(f"Vertices available for selection: {n_vertices}")
    print()

    print("Fields and observed values:")
    for column, display_name, sort_mode in SELECTOR_INVENTORY_FIELDS:
        if column not in vertices:
            continue
        print_selector_value_counts(
            vertices=vertices,
            column=column,
            display_name=display_name,
            max_values=max_values,
            sort_mode=sort_mode,
        )

    print_residue_id_summary(vertices, max_ranges=max_values)
    print_vertex_id_summary(vertices)
    print()
    print("Compute selector examples for this output:")
    for example in selector_examples_for_output(vertices):
        print(f"  {example}")
    print()
    print("Generic selector syntax:")
    print(indent(selector_help_text().rstrip(), "  "))


def print_selector_value_counts(
    vertices: dict[str, list[Any]],
    column: str,
    display_name: str,
    max_values: int | None,
    sort_mode: str,
) -> None:
    values = [value for value in vertices[column] if value not in (None, "")]
    if not values:
        return

    counts = Counter(values)
    total_unique = len(counts)

    print(f"  {display_name}:")
    items = sorted_count_items(counts, sort_mode=sort_mode)
    shown_items = items if max_values is None else items[:max_values]

    for value, count in shown_items:
        selector_value = selector_literal(value)
        print(f"    {column} {selector_value:<24s} {count:8d} vertices")

    if max_values is not None:
        remaining = total_unique - max_values
        if remaining > 0:
            print(f"    ... {remaining} more values; use --long to show all")
    print()


def sorted_count_items(counts: Counter[Any], sort_mode: str) -> list[tuple[Any, int]]:
    items = list(counts.items())
    if sort_mode == "value":
        return sorted(items, key=lambda item: value_sort_key(item[0]))
    return sorted(items, key=lambda item: (-item[1], value_sort_key(item[0])))


def value_sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int):
        return (0, value)
    if isinstance(value, float):
        return (0, value)
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def selector_literal(value: Any) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(char.isspace() for char in text) or any(char in text for char in '()"'):
        escaped = text.replace('"', '\\"')
        return f'"{escaped}"'
    return text


def print_residue_id_summary(vertices: dict[str, list[Any]], max_ranges: int | None = 10) -> None:
    values = numeric_values(vertices.get("residue_id", []))
    if not values:
        return

    print("  residue_id / resid:")
    print(
        f"    unique={len(set(values))}, min={min(values)}, max={max(values)}"
    )
    ranges = contiguous_ranges(sorted(set(values)))
    shown_ranges = ranges if max_ranges is None else ranges[:max_ranges]
    if shown_ranges:
        rendered = ", ".join(render_range(start, stop) for start, stop in shown_ranges)
        suffix = ""
        if max_ranges is not None and len(ranges) > max_ranges:
            suffix = f", ... {len(ranges) - max_ranges} more ranges; use --long to show all"
        print(f"    ranges: {rendered}{suffix}")
    print()


def print_vertex_id_summary(vertices: dict[str, list[Any]]) -> None:
    values = numeric_values(vertices.get("vertex_id", []))
    if not values:
        return
    print("  vertex_id / id:")
    print(f"    min={min(values)}, max={max(values)}, count={len(values)}")
    print()


def numeric_values(values: Iterable[Any]) -> list[int]:
    out: list[int] = []
    for value in values:
        if value in (None, ""):
            continue
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def contiguous_ranges(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    ranges: list[tuple[int, int]] = []
    start = values[0]
    previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = value
        previous = value
    ranges.append((start, previous))
    return ranges


def render_range(start: int, stop: int) -> str:
    if start == stop:
        return str(start)
    return f"{start}-{stop}"


def selector_examples_for_output(vertices: dict[str, list[Any]]) -> list[str]:
    """Return copyable selector examples that are valid for this output.

    The examples are intentionally data-dependent: templates that cannot be
    populated from the current vertex table are omitted rather than shown as
    generic placeholders.
    """
    examples: list[str] = []

    molecule_instances = sorted_unique_values(vertices.get("molecule_instance", []))
    molecule_types = sorted_unique_values(vertices.get("molecule_type", []), by_count=True)
    residue_names = sorted_unique_values(vertices.get("residue_name", []), by_count=True)
    kinds = sorted_unique_values(vertices.get("kind", []), by_count=True)
    vertex_ids = numeric_values(vertices.get("vertex_id", []))

    append_unique(examples, single_value_example("molecule_instance", molecule_instances))
    append_unique(examples, in_values_example("molecule_instance", molecule_instances, n_values=3))
    append_unique(examples, single_value_example("molecule_type", molecule_types))
    append_unique(examples, in_values_example("molecule_type", molecule_types, n_values=2))
    append_unique(examples, single_value_example("resname", residue_names))
    append_unique(examples, in_values_example("resname", residue_names, n_values=3))
    append_unique(examples, single_value_example("kind", kinds))
    append_unique(examples, in_values_example("kind", kinds, n_values=2))

    residue_example = residue_name_and_id_example(vertices)
    append_unique(examples, residue_example)

    molecule_range_example = molecule_residue_range_example(vertices)
    append_unique(examples, molecule_range_example)

    same_molecule_resname_example = same_molecule_resname_or_example(vertices)
    append_unique(examples, same_molecule_resname_example)

    if "bundle" in {str(value) for value in kinds}:
        append_unique(examples, 'not kind bundle')

    vertex_range_example = vertex_id_range_example(vertex_ids)
    append_unique(examples, vertex_range_example)

    return [f'--target "{example}"' for example in examples if example]


def append_unique(examples: list[str], example: str | None) -> None:
    if example and example not in examples:
        examples.append(example)


def sorted_unique_values(values: Iterable[Any], *, by_count: bool = False) -> list[Any]:
    clean = [value for value in values if value not in (None, "")]
    if not clean:
        return []

    counts = Counter(clean)
    if by_count:
        return [value for value, _count in sorted_count_items(counts, sort_mode="count")]
    return [value for value, _count in sorted_count_items(counts, sort_mode="value")]


def single_value_example(field: str, values: list[Any]) -> str | None:
    if not values:
        return None
    return f"{field} {selector_literal(values[0])}"


def in_values_example(field: str, values: list[Any], n_values: int) -> str | None:
    if len(values) < n_values:
        return None
    rendered = ",".join(selector_literal(value) for value in values[:n_values])
    return f"{field} in {rendered}"


def residue_name_and_id_example(vertices: dict[str, list[Any]]) -> str | None:
    residue_names = vertices.get("residue_name", [])
    residue_ids = vertices.get("residue_id", [])
    n_rows = min(len(residue_names), len(residue_ids))

    for index in range(n_rows):
        residue_name = residue_names[index]
        residue_id = residue_ids[index]
        if residue_name in (None, "") or residue_id in (None, ""):
            continue
        return f"resname {selector_literal(residue_name)} and resid {selector_literal(residue_id)}"

    return None


def molecule_residue_range_example(vertices: dict[str, list[Any]]) -> str | None:
    molecule_instances = vertices.get("molecule_instance", [])
    residue_ids = vertices.get("residue_id", [])
    n_rows = min(len(molecule_instances), len(residue_ids))

    by_molecule: dict[Any, set[int]] = {}
    for index in range(n_rows):
        molecule_instance = molecule_instances[index]
        residue_id = residue_ids[index]
        if molecule_instance in (None, "") or residue_id in (None, ""):
            continue
        try:
            residue_id_int = int(residue_id)
        except (TypeError, ValueError):
            continue
        by_molecule.setdefault(molecule_instance, set()).add(residue_id_int)

    candidates = sorted(
        by_molecule.items(),
        key=lambda item: (-len(item[1]), value_sort_key(item[0])),
    )

    for molecule_instance, residue_set in candidates:
        if len(residue_set) < 2:
            continue
        residue_values = sorted(residue_set)
        start, stop = short_residue_window(residue_values, max_width=20)
        if start == stop:
            continue
        return (
            f"molecule_instance {selector_literal(molecule_instance)} "
            f"and resid {start}-{stop}"
        )

    return None


def short_residue_window(residue_values: list[int], max_width: int) -> tuple[int, int]:
    if len(residue_values) < 2:
        value = residue_values[0]
        return value, value

    # Prefer an actual contiguous run, because it produces a clean selector.
    runs = contiguous_ranges(residue_values)
    runs = sorted(runs, key=lambda pair: (-(pair[1] - pair[0]), pair[0]))
    for start, stop in runs:
        if stop > start:
            return start, min(stop, start + max_width - 1)

    # If residues are sparse, fall back to the first two observed residue ids.
    return residue_values[0], residue_values[1]


def same_molecule_resname_or_example(vertices: dict[str, list[Any]]) -> str | None:
    molecule_instances = vertices.get("molecule_instance", [])
    residue_names = vertices.get("residue_name", [])
    n_rows = min(len(molecule_instances), len(residue_names))

    by_molecule: dict[Any, Counter[Any]] = {}
    for index in range(n_rows):
        molecule_instance = molecule_instances[index]
        residue_name = residue_names[index]
        if molecule_instance in (None, "") or residue_name in (None, ""):
            continue
        by_molecule.setdefault(molecule_instance, Counter())[residue_name] += 1

    candidates = sorted(
        by_molecule.items(),
        key=lambda item: (-len(item[1]), value_sort_key(item[0])),
    )

    for molecule_instance, counts in candidates:
        if len(counts) < 2:
            continue
        names = [value for value, _count in sorted_count_items(counts, sort_mode="count")[:2]]
        return (
            f"(resname {selector_literal(names[0])} or "
            f"resname {selector_literal(names[1])}) and "
            f"molecule_instance {selector_literal(molecule_instance)}"
        )

    return None


def vertex_id_range_example(vertex_ids: list[int]) -> str | None:
    if len(vertex_ids) < 2:
        return None
    vertex_ids = sorted(set(vertex_ids))
    start = vertex_ids[0]
    stop = vertex_ids[min(len(vertex_ids) - 1, 9)]
    if start == stop:
        return None
    return f"vertex_id {start}-{stop}"

def first_nonempty(values: Iterable[Any]) -> Any | None:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def validate_output_dir(input_dir: Path) -> None:
    required = [
        input_dir / "metadata.json",
        input_dir / "vertices.parquet",
        input_dir / "edges.parquet",
        input_dir / "vertex_members.parquet",
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"missing redgewise output file: {path}")

    values_dir = input_dir / "values"
    if not values_dir.exists():
        raise FileNotFoundError(f"missing redgewise values directory: {values_dir}")


def read_metadata(input_dir: Path) -> dict[str, Any]:
    with (input_dir / "metadata.json").open() as handle:
        return json.load(handle)


def read_vertex_member_counts(input_dir: Path) -> dict[int, int]:
    table = pq.read_table(input_dir / "vertex_members.parquet", columns=["vertex_id"])
    vertex_ids = table.column("vertex_id").to_pylist()
    counts = Counter(int(vertex_id) for vertex_id in vertex_ids)
    return dict(counts)


def count_edges(input_dir: Path) -> int:
    return pq.read_table(input_dir / "edges.parquet", columns=["edge_key"]).num_rows


def count_value_rows(input_dir: Path) -> int:
    total = 0
    for path in sorted((input_dir / "values").glob("*.parquet")):
        total += pq.read_table(path, columns=["edge_key"]).num_rows
    return total


def count_frames(input_dir: Path) -> int:
    frames: set[int] = set()
    for path in sorted((input_dir / "values").glob("*.parquet")):
        table = pq.read_table(path, columns=["frame"])
        frames.update(int(frame) for frame in table.column("frame").to_pylist())
    return len(frames)


def metadata_value(metadata: dict[str, Any], key: str, fallback: Any) -> Any:
    value = metadata.get(key)
    if value is None:
        return fallback
    return value


def kind_sort_key(kind: str) -> tuple[int, str]:
    return KIND_PRECEDENCE.get(kind, 99), kind



def truncate_vertex_rows(
    rows: list[dict[str, Any]],
    max_per_kind: int = 40,
    max_total: int = 200,
) -> list[dict[str, Any]]:
    """Return a compact but representative subset of sorted vertex rows."""
    selected: list[dict[str, Any]] = []
    counts_by_kind: Counter[str] = Counter()

    for row in rows:
        kind = str(row.get("kind", ""))
        if counts_by_kind[kind] >= max_per_kind:
            continue
        if len(selected) >= max_total:
            break
        selected.append(row)
        counts_by_kind[kind] += 1

    return selected


def iter_sorted_vertex_rows(
    vertices: dict[str, list[Any]],
    vertex_member_counts: dict[int, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    n_vertices = len(vertices["vertex_id"])

    for index in range(n_vertices):
        vertex_id = int(vertices["vertex_id"][index])
        kind = string_value(vertices, "kind", index)

        rows.append(
            {
                "vertex_id": vertex_id,
                "kind": kind,
                "n_atoms": int(vertex_member_counts.get(vertex_id, 0)),
                "residue_name": string_value(vertices, "residue_name", index),
                "residue_id": optional_int(vertices, "residue_id", index),
                "molecule_type": string_value(vertices, "molecule_type", index),
                "molecule_instance": optional_int(vertices, "molecule_instance", index),
                "atom_nr": optional_int(vertices, "atom_nr", index),
                "atom_name": string_value(vertices, "atom_name", index),
                "label": string_value(vertices, "label", index),
            }
        )

    rows.sort(
        key=lambda row: (
            kind_sort_key(row["kind"]),
            row["molecule_instance"] if row["molecule_instance"] is not None else -1,
            row["residue_id"] if row["residue_id"] is not None else -1,
            row["atom_nr"] if row["atom_nr"] is not None else -1,
            row["vertex_id"],
        )
    )

    return rows


def string_value(vertices: dict[str, list[Any]], column: str, index: int) -> str:
    values = vertices.get(column)
    if values is None:
        return ""

    value = values[index]
    if value is None:
        return ""

    return str(value)


def optional_int(vertices: dict[str, list[Any]], column: str, index: int) -> int | None:
    values = vertices.get(column)
    if values is None:
        return None

    value = values[index]
    if value is None:
        return None

    return int(value)


def format_vertex_header() -> str:
    return (
        "  "
        f"{'vertex_id':>9s}  "
        f"{'kind':<14s}  "
        f"{'n_atoms':>7s}  "
        f"{'resname':<8s}  "
        f"{'resid':>7s}  "
        f"{'molinst':>7s}  "
        f"{'atom':>8s}  "
        "label"
    )


def format_vertex_row(row: dict[str, Any]) -> str:
    resid = "" if row["residue_id"] is None else str(row["residue_id"])
    molinst = "" if row["molecule_instance"] is None else str(row["molecule_instance"])
    atom = ""
    if row["atom_nr"] is not None or row["atom_name"]:
        atom_nr = "" if row["atom_nr"] is None else str(row["atom_nr"])
        atom = f"{row['atom_name']}:{atom_nr}" if row["atom_name"] else atom_nr

    return (
        "  "
        f"{row['vertex_id']:9d}  "
        f"{KIND_DISPLAY.get(row['kind'], row['kind']):<14s}  "
        f"{row['n_atoms']:7d}  "
        f"{row['residue_name']:<8.8s}  "
        f"{resid:>7.7s}  "
        f"{molinst:>7.7s}  "
        f"{atom:>8.8s}  "
        f"{row['label']}"
    )
