from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class RedgewiseMdpError(Exception):
    """Expected MDP parsing error with user-readable message."""

class RedgewiseBuildError(Exception):
    """Expected topology/build error with a user-readable message."""

@dataclass(frozen=True)
class MdpNonbondedInformation:
    rlist: float | None
    rcoulomb: float
    rvdw: float
    coulombtype: str | None
    vdwtype: str | None
    rvdw_switch: float | None
    coulomb_modifier: str | None
    vdw_modifier: str | None

    @property
    def max_cutoff(self) -> float:
        return max(self.rcoulomb, self.rvdw)

    @property
    def has_vdw_switch(self) -> bool:
        return self.rvdw_switch is not None

    @property
    def vdw_switch_active(self) -> bool:
        if self.rvdw_switch is None:
            return False
        return self.rvdw_switch < self.rvdw


def read_mdp_nonbonded_information(mdp: Path) -> MdpNonbondedInformation:
    mdp = mdp.expanduser().resolve()

    if not mdp.exists():
        raise RedgewiseMdpError(f"MDP file does not exist: {mdp}")

    values = parse_mdp_file(mdp)

    rcoulomb = get_required_float(values, "rcoulomb", mdp)
    rvdw = get_required_float(values, "rvdw", mdp)

    return MdpNonbondedInformation(
        rlist=get_optional_float(values, "rlist", mdp),
        rcoulomb=rcoulomb,
        rvdw=rvdw,
        coulombtype=values.get("coulombtype"),
        vdwtype=values.get("vdwtype"),
        rvdw_switch=get_optional_float(values, "rvdw-switch", mdp),
        coulomb_modifier=values.get("coulomb-modifier"),
        vdw_modifier=values.get("vdw-modifier"),
    )


def parse_mdp_file(mdp: Path) -> dict[str, str]:
    values: dict[str, str] = {}

    for raw_line in mdp.read_text().splitlines():
        line = raw_line.split(";", 1)[0].strip()

        if not line:
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)

        key = key.strip().lower()
        value = value.strip()

        if not key:
            continue

        values[key] = value

    return values


def get_required_float(
    values: dict[str, str],
    key: str,
    mdp: Path,
) -> float:
    value = values.get(key)

    if value is None:
        raise RedgewiseMdpError(f"required MDP option missing: {key} in {mdp}")

    try:
        return float(value.split()[0])
    except ValueError as exc:
        raise RedgewiseMdpError(
            f"could not parse MDP option {key!r} as float: {value!r}"
        ) from exc


def get_optional_float(
    values: dict[str, str],
    key: str,
    mdp: Path,
) -> float | None:
    value = values.get(key)

    if value is None:
        return None

    try:
        return float(value.split()[0])
    except ValueError as exc:
        raise RedgewiseMdpError(
            f"could not parse MDP option {key!r} as float: {value!r}"
        ) from exc