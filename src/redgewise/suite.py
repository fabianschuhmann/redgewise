from __future__ import annotations

import csv
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from redgewise.analysis_values import canonical_normalization, canonical_value_name
except Exception:  # pragma: no cover - suite should remain importable during development.
    canonical_normalization = None
    canonical_value_name = None

try:
    from redgewise.selectors import SelectorError, evaluate_vertex_selector, vertex_records_to_columns
except Exception:  # pragma: no cover
    SelectorError = ValueError
    evaluate_vertex_selector = None
    vertex_records_to_columns = None


_VALUE_COLUMN_MAP: dict[str, tuple[str, ...]] = {
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

_NORMALIZATIONS = {
    "none",
    "per_atom_pair",
    "per_vertex_member_sqrt",
    "per_vertex_member_product",
    "per_coarse_member_sqrt",
    "per_coarse_member_product",
}


def load_network(path: str | Path) -> RedgewiseNetwork:
    """Load a redgewise build output directory.

    Example
    -------
    >>> from redgewise import suite
    >>> net = suite.load_network("/tmp/output")
    >>> net.metadata["network_directed"]
    False
    """

    return RedgewiseNetwork(Path(path))


def load_tsv(path: str | Path) -> PlotTable:
    """Load a TSV file produced by a redgewise plot command.

    The returned object is a small typed table wrapper backed by NumPy arrays,
    not pandas.
    """

    return PlotTable.from_tsv(path)


def load_plot_table(path: str | Path) -> PlotTable:
    """Alias for :func:`load_tsv`."""

    return load_tsv(path)


def load_plot_tables(paths: str | Path | Iterable[str | Path]) -> dict[str, PlotTable]:
    """Load multiple TSV plot tables.

    `paths` may be a glob pattern, one path, or an iterable of paths. Returned
    keys are file stems.
    """

    if isinstance(paths, (str, Path)):
        text = str(paths)
        if any(char in text for char in "*?[]"):
            resolved = [Path(item) for item in sorted(glob.glob(text))]
        else:
            resolved = [Path(paths)]
    else:
        resolved = [Path(item) for item in paths]

    return {path.stem: load_tsv(path) for path in resolved}


def list_plot_tables(path: str | Path) -> list[Path]:
    """Return sorted TSV files below a directory or the single TSV path."""

    base = Path(path)
    if base.is_file():
        return [base] if base.suffix.lower() == ".tsv" else []
    return sorted(base.glob("*.tsv"))


@dataclass(frozen=True)
class Region:
    label: str
    selector: str
    vertex_ids: np.ndarray
    residue_keys: tuple[tuple[Any, Any, Any], ...]


class SimpleTable:
    """A lightweight columnar table for notebooks.

    Columns are stored as NumPy arrays and can be accessed by name:

    >>> table["frame"]
    >>> table.column("value")
    >>> table.head(5).to_dicts()

    This deliberately avoids pandas while keeping common notebook workflows
    convenient.
    """

    def __init__(
        self,
        columns: Mapping[str, Sequence[Any] | np.ndarray],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        arrays = {str(name): _as_array(values) for name, values in columns.items()}
        lengths = {len(values) for values in arrays.values()}
        if len(lengths) > 1:
            raise ValueError(f"column lengths differ: {sorted(lengths)}")
        self._columns = arrays
        self.metadata = dict(metadata or {})

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._columns.keys())

    @property
    def n_rows(self) -> int:
        if not self._columns:
            return 0
        first = next(iter(self._columns.values()))
        return int(len(first))

    def __len__(self) -> int:
        return self.n_rows

    def __getitem__(self, name: str) -> np.ndarray:
        return self._columns[name]

    def __contains__(self, name: str) -> bool:
        return name in self._columns

    def __repr__(self) -> str:
        cols = ", ".join(self.columns[:8])
        if len(self.columns) > 8:
            cols += ", ..."
        return f"{self.__class__.__name__}(n_rows={self.n_rows}, columns=[{cols}])"

    def column(self, name: str) -> np.ndarray:
        return self._columns[name]

    def get(self, name: str, default: Any = None) -> np.ndarray | Any:
        return self._columns.get(name, default)

    def select(self, names: Iterable[str]) -> SimpleTable:
        selected = {name: self._columns[name] for name in names}
        return self.__class__(selected, metadata=self.metadata)

    def take(self, indices: Sequence[int] | np.ndarray) -> SimpleTable:
        idx = np.asarray(indices)
        return self.__class__(
            {name: values[idx] for name, values in self._columns.items()},
            metadata=self.metadata,
        )

    def where(self, mask: Sequence[bool] | np.ndarray) -> SimpleTable:
        mask_array = np.asarray(mask, dtype=bool)
        if len(mask_array) != self.n_rows:
            raise ValueError(f"mask length {len(mask_array)} does not match table length {self.n_rows}")
        return self.take(mask_array)

    def head(self, n: int = 10) -> SimpleTable:
        return self.take(np.arange(min(max(int(n), 0), self.n_rows)))

    def to_dicts(self, limit: int | None = None) -> list[dict[str, Any]]:
        n = self.n_rows if limit is None else min(int(limit), self.n_rows)
        rows: list[dict[str, Any]] = []
        for index in range(n):
            rows.append({name: _python_scalar(values[index]) for name, values in self._columns.items()})
        return rows

    def unique(self, name: str) -> np.ndarray:
        return np.unique(self._columns[name])

    def to_arrow(self) -> pa.Table:
        return pa.table({name: pa.array(values.tolist()) for name, values in self._columns.items()})

    def write_tsv(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(self.columns)
            for row_index in range(self.n_rows):
                writer.writerow([_python_scalar(self._columns[name][row_index]) for name in self.columns])
        return out


class PlotTable(SimpleTable):
    """A table loaded from a plot TSV file."""

    @classmethod
    def from_tsv(cls, path: str | Path) -> PlotTable:
        source = Path(path)
        with source.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError(f"TSV has no header: {source}")
            raw_columns: dict[str, list[str]] = {name: [] for name in reader.fieldnames}
            for row in reader:
                for name in reader.fieldnames:
                    raw_columns[name].append(row.get(name, ""))

        return cls(
            {name: _infer_tsv_column(values) for name, values in raw_columns.items()},
            metadata={"path": str(source)},
        )


class EdgeValueTable(SimpleTable):
    """Frame-wise selected edge values returned by `RedgewiseNetwork.edge_values`."""


class RedgewiseNetwork:
    """Notebook API for a redgewise build output directory.

    This object is intentionally thin: it reads the canonical build artifacts,
    exposes them as Arrow tables or NumPy-backed `SimpleTable` objects, and
    delegates selector semantics to `redgewise.selectors`.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)

        self._metadata: dict[str, Any] | None = None
        self._vertices_arrow: pa.Table | None = None
        self._members_arrow: pa.Table | None = None
        self._edges_arrow: pa.Table | None = None
        self._vertices: SimpleTable | None = None
        self._edges: SimpleTable | None = None
        self._vertex_records_cache: list[dict[str, Any]] | None = None

    def __repr__(self) -> str:
        try:
            n_vertices = len(self.vertices)
            n_edges = len(self.edges)
        except Exception:
            n_vertices = "?"
            n_edges = "?"
        return f"RedgewiseNetwork(path={str(self.path)!r}, n_vertices={n_vertices}, n_edges={n_edges})"

    @property
    def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            metadata_path = self.path / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(metadata_path)
            self._metadata = json.loads(metadata_path.read_text())
        return self._metadata

    @property
    def values_dir(self) -> Path:
        return self.path / "values"

    @property
    def value_parts(self) -> list[Path]:
        if not self.values_dir.exists():
            return []
        return sorted(self.values_dir.glob("*.parquet"))

    @property
    def vertices_arrow(self) -> pa.Table:
        if self._vertices_arrow is None:
            self._vertices_arrow = pq.read_table(self.path / "vertices.parquet")
        return self._vertices_arrow

    @property
    def vertex_members_arrow(self) -> pa.Table:
        if self._members_arrow is None:
            self._members_arrow = pq.read_table(self.path / "vertex_members.parquet")
        return self._members_arrow

    @property
    def edges_arrow(self) -> pa.Table:
        if self._edges_arrow is None:
            self._edges_arrow = pq.read_table(self.path / "edges.parquet")
        return self._edges_arrow

    @property
    def vertices(self) -> SimpleTable:
        if self._vertices is None:
            self._vertices = _simple_table_from_arrow(self.vertices_arrow, metadata={"path": str(self.path / "vertices.parquet")})
        return self._vertices

    @property
    def edges(self) -> SimpleTable:
        if self._edges is None:
            self._edges = _simple_table_from_arrow(self.edges_arrow, metadata={"path": str(self.path / "edges.parquet")})
        return self._edges

    @property
    def vertex_members(self) -> SimpleTable:
        return _simple_table_from_arrow(
            self.vertex_members_arrow,
            metadata={"path": str(self.path / "vertex_members.parquet")},
        )

    def summary(self) -> dict[str, Any]:
        frames = self.frames()
        return {
            "path": str(self.path),
            "n_vertices": len(self.vertices),
            "n_edges": len(self.edges),
            "n_value_parts": len(self.value_parts),
            "n_frames": len(frames),
            "first_frame": int(frames[0]) if len(frames) else None,
            "last_frame": int(frames[-1]) if len(frames) else None,
            "network_directed": self.metadata.get("network_directed"),
            "edge_values_are_sparse": self.metadata.get("edge_values_are_sparse"),
        }

    def frames(self) -> np.ndarray:
        frame_chunks: list[np.ndarray] = []
        for path in self.value_parts:
            table = pq.read_table(path, columns=["frame"])
            frame_chunks.append(table.column("frame").to_numpy(zero_copy_only=False))
        if not frame_chunks:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate(frame_chunks).astype(np.int64, copy=False))

    def vertex_records(self) -> list[dict[str, Any]]:
        if self._vertex_records_cache is not None:
            return self._vertex_records_cache

        table = self.vertices
        records: list[dict[str, Any]] = []
        for row_index in range(len(table)):
            records.append({name: _python_scalar(table[name][row_index]) for name in table.columns})
        self._vertex_records_cache = records
        return records

    def select_vertices(self, selector: str, *, as_mask: bool = False) -> np.ndarray:
        """Select vertices with the shared redgewise selector grammar.

        Returns vertex IDs by default. Use `as_mask=True` for a boolean mask in
        vertex-table order.
        """

        if evaluate_vertex_selector is None or vertex_records_to_columns is None:
            raise RuntimeError("redgewise.selectors is not available")

        records = self.vertex_records()
        columns = vertex_records_to_columns(records)
        mask = np.asarray(evaluate_vertex_selector(selector, columns, n_rows=len(records)), dtype=bool)
        if len(mask) != len(records):
            raise ValueError(f"selector returned mask length {len(mask)}, expected {len(records)}")
        if as_mask:
            return mask

        vertex_ids = np.asarray(self.vertices["vertex_id"], dtype=np.int64)
        return vertex_ids[mask]

    def regions(
        self,
        selectors: Sequence[str],
        *,
        labels: Sequence[str] | None = None,
        require_disjoint: bool = True,
    ) -> list[Region]:
        if labels is None:
            region_labels = [f"region_{index + 1}" for index in range(len(selectors))]
        else:
            if len(labels) != len(selectors):
                raise ValueError("number of labels must match number of selectors")
            region_labels = [str(label) for label in labels]

        regions: list[Region] = []
        seen: dict[int, str] = {}
        for selector, label in zip(selectors, region_labels):
            ids = np.asarray(self.select_vertices(selector), dtype=np.int64)
            if len(ids) == 0:
                raise ValueError(f"region {label!r} matched zero vertices: {selector!r}")

            if require_disjoint:
                overlap = sorted(int(vertex_id) for vertex_id in ids if int(vertex_id) in seen)
                if overlap:
                    previous = seen[overlap[0]]
                    raise ValueError(
                        f"region {label!r} overlaps with region {previous!r}; first overlapping vertex_id={overlap[0]}"
                    )
                for vertex_id in ids:
                    seen[int(vertex_id)] = label

            regions.append(
                Region(
                    label=label,
                    selector=str(selector),
                    vertex_ids=ids,
                    residue_keys=self.residue_keys_for_vertices(ids),
                )
            )

        return regions

    def residue_keys_for_vertices(self, vertex_ids: Sequence[int] | np.ndarray) -> tuple[tuple[Any, Any, Any], ...]:
        vertex_id_array = np.asarray(self.vertices["vertex_id"], dtype=np.int64)
        order = np.argsort(vertex_id_array)
        sorted_ids = vertex_id_array[order]

        requested = np.asarray(vertex_ids, dtype=np.int64)
        positions = np.searchsorted(sorted_ids, requested)
        valid = (positions >= 0) & (positions < len(sorted_ids)) & (sorted_ids[positions] == requested)
        if not np.all(valid):
            missing = requested[~valid]
            raise KeyError(f"unknown vertex IDs: {missing[:10].tolist()}")

        rows = order[positions]
        vertices = self.vertices
        keys: list[tuple[Any, Any, Any]] = []
        for row in rows:
            keys.append(
                (
                    _python_scalar(vertices.get("molecule_instance", np.full(len(vertices), None, dtype=object))[row]),
                    _python_scalar(vertices.get("residue_id", np.full(len(vertices), None, dtype=object))[row]),
                    _python_scalar(vertices.get("residue_name", np.full(len(vertices), None, dtype=object))[row]),
                )
            )
        return tuple(keys)

    def member_counts(self) -> np.ndarray:
        vertices = self.vertices
        vertex_ids = np.asarray(vertices["vertex_id"], dtype=np.int64)
        n_vertices = int(vertex_ids.max()) + 1 if len(vertex_ids) else 0

        if "members" in vertices:
            raw = np.asarray(vertices["members"])
            counts = np.ones(n_vertices, dtype=np.float64)
            for row, vertex_id in enumerate(vertex_ids):
                try:
                    value = float(raw[row])
                    if math.isfinite(value) and value > 0.0:
                        counts[int(vertex_id)] = value
                except Exception:
                    pass
            return counts

        members = self.vertex_members
        counts = np.bincount(np.asarray(members["vertex_id"], dtype=np.int64), minlength=n_vertices).astype(np.float64)
        counts[counts <= 0.0] = 1.0
        return counts

    def edge_values(
        self,
        value: str = "vdw+cl",
        *,
        normalization: str = "none",
        frames: Iterable[int] | None = None,
        columns: Sequence[str] | None = None,
    ) -> EdgeValueTable:
        """Load selected frame-wise edge values.

        Missing sparse rows are not densified. Missing frame-edge rows still
        semantically mean zero in the build output.

        Parameters
        ----------
        value
            `vdw`, `cl`, `coulomb`, `vdw+cl`, `dvdw`, `dcl`, or `dvdw+dcl`.
        normalization
            Same normalization names used by plotting commands.
        frames
            Optional frame IDs to keep.
        columns
            Optional additional raw columns to retain when present.
        """

        value_name = _canonical_value(value)
        normalization_name = _canonical_norm(normalization)
        value_columns = _value_columns(value_name)

        requested = ["frame", "edge_key", *value_columns, "n_atom_pairs"]
        if columns is not None:
            requested.extend(str(name) for name in columns)
        requested = _unique_preserving_order(requested)

        frame_filter = None if frames is None else set(int(frame) for frame in frames)

        out_columns: dict[str, list[np.ndarray]] = {
            "frame": [],
            "edge_key": [],
            "vertex1": [],
            "vertex2": [],
            "value_raw": [],
            "value": [],
            "n_atom_pairs": [],
        }

        for path in self.value_parts:
            available = set(pq.read_schema(path).names)
            read_columns = [name for name in requested if name in available]
            missing_value_columns = [name for name in value_columns if name not in available]
            if missing_value_columns:
                raise ValueError(f"{path} is missing value columns required for {value_name}: {missing_value_columns}")

            table = pq.read_table(path, columns=read_columns)
            part = _simple_table_from_arrow(table)

            part_frames = np.asarray(part["frame"], dtype=np.int64)
            if frame_filter is not None:
                keep = np.array([int(frame) in frame_filter for frame in part_frames], dtype=bool)
                if not np.any(keep):
                    continue
                part = part.where(keep)
                part_frames = np.asarray(part["frame"], dtype=np.int64)

            edge_keys = np.asarray(part["edge_key"], dtype=np.int64)
            vertex1, vertex2 = self.edge_vertices_for_keys(edge_keys)

            raw = np.zeros(len(part), dtype=np.float64)
            for name in value_columns:
                raw += np.asarray(part[name], dtype=np.float64)

            n_atom_pairs = (
                np.asarray(part["n_atom_pairs"], dtype=np.float64)
                if "n_atom_pairs" in part
                else np.ones(len(part), dtype=np.float64)
            )
            normalized = self._normalize_values(raw, vertex1, vertex2, n_atom_pairs, normalization_name)

            out_columns["frame"].append(part_frames)
            out_columns["edge_key"].append(edge_keys)
            out_columns["vertex1"].append(vertex1)
            out_columns["vertex2"].append(vertex2)
            out_columns["value_raw"].append(raw)
            out_columns["value"].append(normalized)
            out_columns["n_atom_pairs"].append(n_atom_pairs)

        concatenated = {
            name: np.concatenate(chunks) if chunks else _empty_array_for_column(name)
            for name, chunks in out_columns.items()
        }

        return EdgeValueTable(
            concatenated,
            metadata={
                "network_path": str(self.path),
                "value": value_name,
                "normalization": normalization_name,
                "sparse": True,
                "missing_edge_values_are_zero": True,
            },
        )

    def edge_vertices_for_keys(self, edge_keys: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keys = np.asarray(edge_keys, dtype=np.int64)
        edge_table = self.edges
        known_keys = np.asarray(edge_table["edge_key"], dtype=np.int64)
        order = np.argsort(known_keys)
        sorted_keys = known_keys[order]
        positions = np.searchsorted(sorted_keys, keys)
        valid = (positions >= 0) & (positions < len(sorted_keys)) & (sorted_keys[positions] == keys)
        if not np.all(valid):
            missing = keys[~valid]
            raise KeyError(f"unknown edge keys: {missing[:10].tolist()}")

        vertex1 = np.asarray(edge_table["vertex1"], dtype=np.int64)[order][positions]
        vertex2 = np.asarray(edge_table["vertex2"], dtype=np.int64)[order][positions]
        return vertex1, vertex2

    def edge_summary(self, value: str = "vdw+cl", *, normalization: str = "none") -> SimpleTable:
        """Return one row per edge with mean sparse value over observed rows.

        This intentionally does not densify missing frame-edge rows. Use the
        plotting analysis layer if you need the exact same dense/missing-zero
        semantics as a specific plot.
        """

        values = self.edge_values(value=value, normalization=normalization)
        if len(values) == 0:
            return SimpleTable(
                {
                    "edge_key": np.array([], dtype=np.int64),
                    "vertex1": np.array([], dtype=np.int64),
                    "vertex2": np.array([], dtype=np.int64),
                    "mean_value": np.array([], dtype=np.float64),
                    "mean_abs_value": np.array([], dtype=np.float64),
                    "n_rows": np.array([], dtype=np.int64),
                },
                metadata=values.metadata,
            )

        edge_keys = np.asarray(values["edge_key"], dtype=np.int64)
        unique_keys, inverse = np.unique(edge_keys, return_inverse=True)
        sums = np.bincount(inverse, weights=np.asarray(values["value"], dtype=np.float64))
        abs_sums = np.bincount(inverse, weights=np.abs(np.asarray(values["value"], dtype=np.float64)))
        counts = np.bincount(inverse).astype(np.int64)
        vertex1, vertex2 = self.edge_vertices_for_keys(unique_keys)

        return SimpleTable(
            {
                "edge_key": unique_keys,
                "vertex1": vertex1,
                "vertex2": vertex2,
                "mean_value": sums / counts,
                "mean_abs_value": abs_sums / counts,
                "n_rows": counts,
            },
            metadata=values.metadata,
        )

    def plot_table(self, path: str | Path) -> PlotTable:
        return load_tsv(path)

    def plot_tables(self, pattern: str = "*.tsv") -> dict[str, PlotTable]:
        return load_plot_tables(str(self.path / pattern))

    def _normalize_values(
        self,
        raw: np.ndarray,
        vertex1: np.ndarray,
        vertex2: np.ndarray,
        n_atom_pairs: np.ndarray,
        normalization: str,
    ) -> np.ndarray:
        values = np.asarray(raw, dtype=np.float64).copy()
        if normalization == "none":
            return values

        if normalization == "per_atom_pair":
            denom = np.asarray(n_atom_pairs, dtype=np.float64)
        else:
            counts = self.member_counts()
            c1 = counts[np.asarray(vertex1, dtype=np.int64)]
            c2 = counts[np.asarray(vertex2, dtype=np.int64)]

            if normalization == "per_vertex_member_sqrt":
                denom = np.sqrt(c1 * c2)
            elif normalization == "per_vertex_member_product":
                denom = c1 * c2
            elif normalization in {"per_coarse_member_sqrt", "per_coarse_member_product"}:
                kinds = self._kind_by_vertex_id()
                coarse1 = np.array([_is_coarse_kind(kinds[int(vertex_id)]) for vertex_id in vertex1], dtype=bool)
                coarse2 = np.array([_is_coarse_kind(kinds[int(vertex_id)]) for vertex_id in vertex2], dtype=bool)
                d1 = np.where(coarse1, c1, 1.0)
                d2 = np.where(coarse2, c2, 1.0)
                denom = np.sqrt(d1 * d2) if normalization.endswith("_sqrt") else d1 * d2
            else:
                raise ValueError(f"unsupported normalization: {normalization}")

        out = np.full_like(values, np.nan, dtype=np.float64)
        np.divide(values, denom, out=out, where=np.asarray(denom) > 0.0)
        return out

    def _kind_by_vertex_id(self) -> np.ndarray:
        vertices = self.vertices
        vertex_ids = np.asarray(vertices["vertex_id"], dtype=np.int64)
        n_vertices = int(vertex_ids.max()) + 1 if len(vertex_ids) else 0
        kinds = np.full(n_vertices, "", dtype=object)
        if "kind" in vertices:
            for row, vertex_id in enumerate(vertex_ids):
                kinds[int(vertex_id)] = str(vertices["kind"][row])
        return kinds


def _simple_table_from_arrow(table: pa.Table, metadata: Mapping[str, Any] | None = None) -> SimpleTable:
    columns: dict[str, np.ndarray] = {}
    for name in table.column_names:
        columns[name] = table.column(name).to_numpy(zero_copy_only=False)
    return SimpleTable(columns, metadata=metadata)


def _as_array(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    return np.asarray(list(values))


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode()
    return value


def _infer_tsv_column(values: list[str]) -> np.ndarray:
    non_empty = [value for value in values if value not in {"", "None", "none", "NaN", "nan"}]
    if not non_empty:
        return np.asarray(values, dtype=object)

    if all(_looks_like_int(value) for value in non_empty):
        out = np.empty(len(values), dtype=np.float64 if len(non_empty) != len(values) else np.int64)
        for index, value in enumerate(values):
            if value in {"", "None", "none", "NaN", "nan"}:
                out[index] = np.nan
            else:
                out[index] = int(value)
        return out.astype(np.int64) if len(non_empty) == len(values) else out

    if all(_looks_like_float(value) for value in non_empty):
        out = np.empty(len(values), dtype=np.float64)
        for index, value in enumerate(values):
            if value in {"", "None", "none", "NaN", "nan"}:
                out[index] = np.nan
            else:
                out[index] = float(value)
        return out

    if all(value.lower() in {"true", "false"} for value in non_empty):
        if len(non_empty) == len(values):
            return np.asarray([value.lower() == "true" for value in values], dtype=bool)

    return np.asarray(values, dtype=object)


def _looks_like_int(value: str) -> bool:
    try:
        int(value)
        return "." not in value and "e" not in value.lower()
    except ValueError:
        return False


def _looks_like_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _canonical_value(value: str) -> str:
    if canonical_value_name is not None:
        try:
            return str(canonical_value_name(value))
        except Exception:
            pass
    key = str(value).strip().lower()
    if key not in _VALUE_COLUMN_MAP:
        raise ValueError(f"unknown value name {value!r}; expected one of {sorted(_VALUE_COLUMN_MAP)}")
    if key == "coulomb":
        return "cl"
    if key == "vdw+coulomb":
        return "vdw+cl"
    if key == "dcoulomb":
        return "dcl"
    if key == "dvdw+dcoulomb":
        return "dvdw+dcl"
    return key


def _value_columns(value: str) -> tuple[str, ...]:
    key = _canonical_value(value)
    return _VALUE_COLUMN_MAP[key]


def _canonical_norm(normalization: str) -> str:
    if canonical_normalization is not None:
        try:
            return str(canonical_normalization(normalization))
        except Exception:
            pass
    key = str(normalization).strip().lower()
    if key not in _NORMALIZATIONS:
        raise ValueError(f"unknown normalization {normalization!r}; expected one of {sorted(_NORMALIZATIONS)}")
    return key


def _unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _empty_array_for_column(name: str) -> np.ndarray:
    if name in {"frame", "edge_key", "vertex1", "vertex2"}:
        return np.array([], dtype=np.int64)
    return np.array([], dtype=np.float64)


def _is_coarse_kind(kind: Any) -> bool:
    return str(kind).lower() in {"bundle", "low_res", "coarse"}


__all__ = [
    "EdgeValueTable",
    "PlotTable",
    "RedgewiseNetwork",
    "Region",
    "SimpleTable",
    "list_plot_tables",
    "load_network",
    "load_plot_table",
    "load_plot_tables",
    "load_tsv",
]
