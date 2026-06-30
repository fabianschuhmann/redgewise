from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


class SelectorError(ValueError):
    """User-readable selector parse or evaluation error."""


FIELD_ALIASES: dict[str, str] = {
    "vertex_id": "vertex_id",
    "id": "vertex_id",
    "label": "label",
    "kind": "kind",
    "residue_name": "residue_name",
    "resname": "residue_name",
    "residue_id": "residue_id",
    "resid": "residue_id",
    "molecule_type": "molecule_type",
    "moltype": "molecule_type",
    "molecule_instance": "molecule_instance",
    "molinstance": "molecule_instance",
    "atom_nr": "atom_nr",
    "bynum": "atom_nr",
    "atom_name": "atom_name",
    "name": "atom_name",
    "atom_type": "atom_type",
    "type": "atom_type",
    "charge": "charge",
    "members": "members",
}

NUMERIC_FIELDS = {
    "vertex_id",
    "residue_id",
    "molecule_instance",
    "atom_nr",
    "charge",
    "members",
}

STRING_FIELDS = set(FIELD_ALIASES.values()) - NUMERIC_FIELDS
LOGICAL_TOKENS = {"and", "or"}
COMPARISON_OPERATORS = {"==", "!=", "<", "<=", ">", ">="}
_RANGE_RE = re.compile(r"^(-?\d+)\s*-\s*(-?\d+)$")


@dataclass(frozen=True)
class Predicate:
    field: str
    operator: str
    value: str


@dataclass(frozen=True)
class NotNode:
    child: Any


@dataclass(frozen=True)
class BinaryNode:
    operator: str
    left: Any
    right: Any


class SelectorParser:
    def __init__(self, selector: str):
        self.selector = selector
        self.tokens = tokenize_selector(selector)
        self.index = 0

    def parse(self) -> Any:
        if not self.tokens:
            raise SelectorError("empty selector")
        node = self.parse_or()
        if self.index != len(self.tokens):
            raise SelectorError(
                f"unexpected token in selector {self.selector!r}: {self.tokens[self.index]!r}"
            )
        return node

    def parse_or(self) -> Any:
        node = self.parse_and()
        while self.peek_lower() == "or":
            self.consume()
            node = BinaryNode("or", node, self.parse_and())
        return node

    def parse_and(self) -> Any:
        node = self.parse_factor()
        while self.peek_lower() == "and":
            self.consume()
            node = BinaryNode("and", node, self.parse_factor())
        return node

    def parse_factor(self) -> Any:
        token = self.peek()
        if token is None:
            raise SelectorError(f"unexpected end of selector {self.selector!r}")
        token_lower = token.lower()
        if token_lower == "not":
            self.consume()
            return NotNode(self.parse_factor())
        if token == "(":
            self.consume()
            node = self.parse_or()
            if self.peek() != ")":
                raise SelectorError(f"missing ')' in selector {self.selector!r}")
            self.consume()
            return node
        return self.parse_predicate()

    def parse_predicate(self) -> Predicate:
        field_token = self.consume()
        field = normalize_field_name(field_token)

        next_token = self.peek()
        if next_token is None:
            raise SelectorError(f"missing value after selector field {field_token!r}")

        next_lower = next_token.lower()
        if next_token in COMPARISON_OPERATORS:
            operator = self.consume()
            value = self.consume_value()
        elif next_lower == "in":
            self.consume()
            operator = "in"
            value = self.consume_value()
        else:
            operator = "=="
            value = self.consume_value()

        return Predicate(field=field, operator=operator, value=value)

    def consume_value(self) -> str:
        value = self.consume()
        if value in {"(", ")"} or value.lower() in LOGICAL_TOKENS:
            raise SelectorError(f"expected selector value, got {value!r}")
        return value

    def peek(self) -> str | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def peek_lower(self) -> str | None:
        token = self.peek()
        if token is None:
            return None
        return token.lower()

    def consume(self) -> str:
        token = self.peek()
        if token is None:
            raise SelectorError(f"unexpected end of selector {self.selector!r}")
        self.index += 1
        return token


def tokenize_selector(selector: str) -> list[str]:
    spaced = selector.replace("(", " ( ").replace(")", " ) ")
    try:
        return shlex.split(spaced)
    except ValueError as exc:
        raise SelectorError(f"could not parse selector {selector!r}: {exc}") from exc


def normalize_field_name(name: str) -> str:
    key = name.strip().lower()
    if key not in FIELD_ALIASES:
        known = ", ".join(sorted(FIELD_ALIASES))
        raise SelectorError(f"unknown selector field {name!r}; known fields: {known}")
    return FIELD_ALIASES[key]


def evaluate_vertex_selector(
    selector: str,
    columns: Mapping[str, Sequence[Any]],
    *,
    n_rows: int | None = None,
) -> np.ndarray:
    """Return a boolean mask for rows in a vertex table-like mapping."""
    if n_rows is None:
        n_rows = infer_n_rows(columns)
    ast = SelectorParser(selector).parse()
    mask = evaluate_node(ast, columns, n_rows)
    return np.asarray(mask, dtype=bool)


def infer_n_rows(columns: Mapping[str, Sequence[Any]]) -> int:
    if not columns:
        raise SelectorError("cannot evaluate selector on an empty table")
    first = next(iter(columns.values()))
    return len(first)


def evaluate_node(node: Any, columns: Mapping[str, Sequence[Any]], n_rows: int) -> np.ndarray:
    if isinstance(node, Predicate):
        return evaluate_predicate(node, columns, n_rows)
    if isinstance(node, NotNode):
        return ~evaluate_node(node.child, columns, n_rows)
    if isinstance(node, BinaryNode):
        left = evaluate_node(node.left, columns, n_rows)
        right = evaluate_node(node.right, columns, n_rows)
        if node.operator == "and":
            return left & right
        if node.operator == "or":
            return left | right
    raise SelectorError(f"internal selector parser error: unknown node {node!r}")


def evaluate_predicate(
    predicate: Predicate,
    columns: Mapping[str, Sequence[Any]],
    n_rows: int,
) -> np.ndarray:
    values = column_as_array(columns, predicate.field, n_rows)
    if predicate.field in NUMERIC_FIELDS:
        numeric = as_float_array(values)
        return evaluate_numeric_predicate(numeric, predicate.operator, predicate.value)
    strings = as_string_array(values)
    return evaluate_string_predicate(strings, predicate.operator, predicate.value)


def column_as_array(
    columns: Mapping[str, Sequence[Any]],
    field: str,
    n_rows: int,
) -> np.ndarray:
    if field in columns:
        return np.asarray(columns[field], dtype=object)
    # Missing optional fields evaluate as empty/NaN values rather than failing.
    return np.asarray([None] * n_rows, dtype=object)


def as_float_array(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for i, value in enumerate(values):
        if value is None or value == "":
            continue
        try:
            out[i] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def as_string_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(["" if value is None else str(value) for value in values], dtype=object)


def evaluate_numeric_predicate(values: np.ndarray, operator: str, raw_value: str) -> np.ndarray:
    if operator == "in":
        mask = np.zeros(values.shape, dtype=bool)
        for item in split_value_list(raw_value):
            mask |= evaluate_numeric_predicate(values, "==", item)
        return mask

    range_match = _RANGE_RE.match(raw_value)
    if range_match and operator == "==":
        lower = float(range_match.group(1))
        upper = float(range_match.group(2))
        if lower > upper:
            lower, upper = upper, lower
        return np.isfinite(values) & (values >= lower) & (values <= upper)

    try:
        target = float(raw_value)
    except ValueError as exc:
        raise SelectorError(f"expected numeric selector value, got {raw_value!r}") from exc

    finite = np.isfinite(values)
    if operator == "==":
        return finite & (values == target)
    if operator == "!=":
        return (~finite) | (values != target)
    if operator == "<":
        return finite & (values < target)
    if operator == "<=":
        return finite & (values <= target)
    if operator == ">":
        return finite & (values > target)
    if operator == ">=":
        return finite & (values >= target)
    raise SelectorError(f"unsupported numeric selector operator {operator!r}")


def evaluate_string_predicate(values: np.ndarray, operator: str, raw_value: str) -> np.ndarray:
    if operator == "in":
        allowed = set(split_value_list(raw_value))
        return np.asarray([value in allowed for value in values], dtype=bool)
    if operator == "==":
        return values == raw_value
    if operator == "!=":
        return values != raw_value
    raise SelectorError(f"operator {operator!r} is not supported for string fields")


def split_value_list(value: str) -> list[str]:
    if not value:
        return []
    stripped = value.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        stripped = stripped[1:-1]
    return [item for item in (part.strip() for part in stripped.split(",")) if item]


def vertex_records_to_columns(vertices: Sequence[Any]) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {canonical: [] for canonical in sorted(set(FIELD_ALIASES.values()))}
    for vertex in vertices:
        for field in columns:
            columns[field].append(getattr(vertex, field, None))
    return columns


def selector_help_text() -> str:
    return """Selector syntax:
  SELECTOR is a small boolean expression over vertex-table fields.

Fields:
  vertex_id/id, label, kind, residue_name/resname, residue_id/resid,
  molecule_type/moltype, molecule_instance/molinstance,
  atom_name/name, atom_type/type, atom_nr/bynum, charge, members

Operators:
  and, or, not, parentheses, ==, !=, <, <=, >, >=, in, integer ranges A-B

Examples:
  molecule_instance 0
  molinstance 0
  resname ARG and resid 76
  resid 76-80 and molecule_instance 0
  kind atom and resname ARG
  resname in POPC,POPE,POPS
  not kind bundle
"""
