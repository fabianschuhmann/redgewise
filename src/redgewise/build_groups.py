from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np


SELECTION_OPERATORS = {"and", "or", "not"}
SELECTION_STOP_TOKENS = SELECTION_OPERATORS | {")"}


@dataclass(frozen=True)
class GroupingInformation:
    vertices: list[dict]
    atom_table: dict[str, np.ndarray]
    atom_to_vertex: np.ndarray
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ResolutionInformation:
    high_mask: np.ndarray
    low_id_by_atom: np.ndarray
    low_labels: dict[int, str]
    bundle_id_by_atom: np.ndarray
    bundle_labels: dict[int, str]
    warnings: tuple[str, ...]


def build_grouping_information(
    interaction_information: Any,
    options: Any,
) -> GroupingInformation:
    atom_table = build_atom_table(interaction_information)
    resolution = build_resolution_information(atom_table=atom_table, options=options)

    vertices: list[dict] = []
    vertex_key_to_id: dict[tuple, int] = {}

    n_atoms = len(atom_table["atom_index"])
    atom_to_vertex = np.empty(n_atoms, dtype=np.int32)

    for atom_index in range(n_atoms):
        row = atom_row(atom_table, atom_index)

        vertex_key, vertex_kind, vertex_label = classify_atom_vertex(
            row=row,
            atom_index=atom_index,
            resolution=resolution,
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
        warnings=resolution.warnings,
    )


def build_resolution_information(
    atom_table: dict[str, np.ndarray],
    options: Any,
) -> ResolutionInformation:
    n_atoms = len(atom_table["atom_index"])
    warnings: list[str] = []

    high_selectors = tuple(getattr(options, "high_res", ()))
    low_selectors = tuple(getattr(options, "low_res", ()))
    bundles = tuple(tuple(bundle) for bundle in getattr(options, "bundles", ()))

    high_mask = np.zeros(n_atoms, dtype=bool)
    low_id_by_atom = np.full(n_atoms, -1, dtype=np.int32)
    bundle_id_by_atom = np.full(n_atoms, -1, dtype=np.int32)

    low_labels: dict[int, str] = {}
    bundle_labels: dict[int, str] = {}

    for selector in high_selectors:
        mask = selection_mask(atom_table, selector)
        if not np.any(mask):
            warnings.append(
                f"resolution warning: --high_res selector {selector!r} matched no atoms."
            )
        high_mask |= mask

    for low_id, selector in enumerate(low_selectors):
        mask = selection_mask(atom_table, selector)
        low_labels[low_id] = selector_label(selector)

        if not np.any(mask):
            warnings.append(
                f"resolution warning: --low_res selector {selector!r} matched no atoms."
            )
            continue

        overlap = mask & (low_id_by_atom >= 0)
        if np.any(overlap):
            warnings.append(
                "resolution warning: atoms matched by multiple --low_res selectors; "
                "using the first matching selector for overlaps."
            )

        assign = mask & (low_id_by_atom < 0)
        low_id_by_atom[assign] = low_id

    for bundle_id, bundle in enumerate(bundles):
        bundle_labels[bundle_id] = "+".join(selector_label(selector) for selector in bundle)
        bundle_mask = np.zeros(n_atoms, dtype=bool)

        for selector in bundle:
            mask = selection_mask(atom_table, selector)
            if not np.any(mask):
                warnings.append(
                    "resolution warning: --bundle selector "
                    f"{selector!r} matched no atoms."
                )
            bundle_mask |= mask

        overlap = bundle_mask & (bundle_id_by_atom >= 0)
        if np.any(overlap):
            warnings.append(
                "resolution warning: atoms matched by multiple --bundle definitions; "
                "using the first matching bundle for overlaps."
            )

        assign = bundle_mask & (bundle_id_by_atom < 0)
        bundle_id_by_atom[assign] = bundle_id

    low_mask = low_id_by_atom >= 0
    bundle_mask = bundle_id_by_atom >= 0

    if np.any(high_mask & low_mask):
        warnings.append(
            "resolution warning: some atoms match both --high_res and --low_res; "
            "--high_res takes precedence."
        )

    if np.any(high_mask & bundle_mask):
        warnings.append(
            "resolution warning: some atoms match both --high_res and --bundle; "
            "--high_res takes precedence."
        )

    if np.any(low_mask & bundle_mask):
        warnings.append(
            "resolution warning: some atoms match both --low_res and --bundle; "
            "--low_res takes precedence."
        )

    return ResolutionInformation(
        high_mask=high_mask,
        low_id_by_atom=low_id_by_atom,
        low_labels=low_labels,
        bundle_id_by_atom=bundle_id_by_atom,
        bundle_labels=bundle_labels,
        warnings=tuple(warnings),
    )


def classify_atom_vertex(
    row: dict,
    atom_index: int,
    resolution: ResolutionInformation,
) -> tuple[tuple, str, str]:
    if resolution.high_mask[atom_index]:
        return (
            ("atom", int(row["atom_nr"])),
            "atom",
            (
                f"{row['residue_name']}:{row['residue_id']}:"
                f"{row['atom_name']}:{row['atom_nr']}"
            ),
        )

    low_id = int(resolution.low_id_by_atom[atom_index])
    if low_id >= 0:
        label = resolution.low_labels[low_id]
        return (
            ("low_res", low_id),
            "low_res",
            label,
        )

    bundle_id = int(resolution.bundle_id_by_atom[atom_index])
    if bundle_id >= 0:
        label = resolution.bundle_labels[bundle_id]
        return (
            ("bundle", bundle_id),
            "bundle",
            label,
        )

    return (
        ("residue", int(row["residue_id"])),
        "residue",
        f"{row['residue_name']}:{row['residue_id']}",
    )


def atom_row(atom_table: dict[str, np.ndarray], index: int) -> dict:
    return {key: values[index] for key, values in atom_table.items()}


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


def selection_mask(atom_table: dict[str, np.ndarray], selector: str) -> np.ndarray:
    selector = selector.strip()

    if not selector:
        raise ValueError("empty resolution selector")

    shorthand = shorthand_selection_mask(atom_table, selector)
    if shorthand is not None:
        return shorthand

    tokens = tokenize_selector(selector)
    parser = SelectionParser(tokens=tokens, atom_table=atom_table, selector=selector)
    mask = parser.parse_expression()

    if parser.position != len(tokens):
        raise ValueError(
            "could not parse full resolution selector "
            f"{selector!r}; unexpected token {tokens[parser.position]!r}"
        )

    return mask


def shorthand_selection_mask(
    atom_table: dict[str, np.ndarray],
    selector: str,
) -> np.ndarray | None:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_+-]*", selector):
        return string_column_mask(atom_table["residue_name"], [selector])

    if re.fullmatch(r"\d+", selector):
        return integer_column_mask(atom_table["residue_id"], [selector])

    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_+-]*):(\d+(?:[-:]\d+)?)", selector)
    if match:
        resname, resid = match.groups()
        return string_column_mask(atom_table["residue_name"], [resname]) & integer_column_mask(
            atom_table["residue_id"],
            [resid],
        )

    return None


def tokenize_selector(selector: str) -> list[str]:
    return re.findall(r"\(|\)|[^\s()]+", selector)


class SelectionParser:
    def __init__(
        self,
        tokens: list[str],
        atom_table: dict[str, np.ndarray],
        selector: str,
    ) -> None:
        self.tokens = tokens
        self.atom_table = atom_table
        self.selector = selector
        self.position = 0
        self.n_atoms = len(atom_table["atom_index"])

    def parse_expression(self) -> np.ndarray:
        mask = self.parse_term()

        while self.peek_lower() == "or":
            self.position += 1
            mask = mask | self.parse_term()

        return mask

    def parse_term(self) -> np.ndarray:
        mask = self.parse_factor()

        while self.peek_lower() == "and":
            self.position += 1
            mask = mask & self.parse_factor()

        return mask

    def parse_factor(self) -> np.ndarray:
        token = self.peek()

        if token is None:
            raise ValueError(f"unexpected end of selector {self.selector!r}")

        if token.lower() == "not":
            self.position += 1
            return ~self.parse_factor()

        if token == "(":
            self.position += 1
            mask = self.parse_expression()
            if self.peek() != ")":
                raise ValueError(f"missing ')' in selector {self.selector!r}")
            self.position += 1
            return mask

        return self.parse_predicate()

    def parse_predicate(self) -> np.ndarray:
        keyword = self.consume().lower()
        values: list[str] = []

        while True:
            token = self.peek()
            if token is None or token.lower() in SELECTION_STOP_TOKENS:
                break
            values.append(self.consume())

        if not values:
            raise ValueError(
                f"selection keyword {keyword!r} in {self.selector!r} has no values"
            )

        if keyword == "resname":
            return string_column_mask(self.atom_table["residue_name"], values)

        if keyword in {"resid", "residue", "residue_id"}:
            return integer_column_mask(self.atom_table["residue_id"], values)

        if keyword in {"name", "atomname", "atom_name"}:
            return string_column_mask(self.atom_table["atom_name"], values)

        if keyword in {"type", "atomtype", "atom_type"}:
            return string_column_mask(self.atom_table["atom_type"], values)

        if keyword in {"moltype", "molecule_type"}:
            return string_column_mask(self.atom_table["molecule_type"], values)

        if keyword in {"molinstance", "molecule_instance"}:
            return integer_column_mask(self.atom_table["molecule_instance"], values)

        if keyword in {"index", "atom_index"}:
            return integer_column_mask(self.atom_table["atom_index"], values)

        if keyword in {"bynum", "atomnr", "atom_nr"}:
            return integer_column_mask(self.atom_table["atom_nr"], values)

        raise ValueError(
            "unsupported resolution selector keyword "
            f"{keyword!r} in {self.selector!r}; supported keywords are "
            "resname, resid, name, type, moltype, molinstance, index, bynum."
        )

    def peek(self) -> str | None:
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]

    def peek_lower(self) -> str | None:
        token = self.peek()
        if token is None:
            return None
        return token.lower()

    def consume(self) -> str:
        token = self.tokens[self.position]
        self.position += 1
        return token


def string_column_mask(column: np.ndarray, values: list[str]) -> np.ndarray:
    normalized = np.array([str(value).upper() for value in column], dtype=object)
    wanted = {str(value).upper() for value in values}
    return np.array([value in wanted for value in normalized], dtype=bool)


def integer_column_mask(column: np.ndarray, values: list[str]) -> np.ndarray:
    wanted: set[int] = set()

    for value in values:
        wanted.update(parse_integer_selector_value(value))

    return np.isin(column.astype(np.int64), np.array(sorted(wanted), dtype=np.int64))


def parse_integer_selector_value(value: str) -> set[int]:
    if re.fullmatch(r"\d+", value):
        return {int(value)}

    match = re.fullmatch(r"(\d+)[-:](\d+)", value)
    if match:
        start, stop = map(int, match.groups())
        if stop < start:
            start, stop = stop, start
        return set(range(start, stop + 1))

    raise ValueError(
        f"could not parse integer selector value {value!r}; "
        "expected INT, START-END, or START:END"
    )


def selector_label(selector: str) -> str:
    shorthand = selector.strip()
    if len(shorthand) <= 48:
        return shorthand
    return shorthand[:45] + "..."
