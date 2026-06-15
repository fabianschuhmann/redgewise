from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import MDAnalysis as mda

from redgewise.build_information import (
    InteractionInformation,
    VdwInteraction,
    pair_key,
)


class RedgewiseBuildError(Exception):
    """Expected build-time error with user-readable message."""


@dataclass(frozen=True)
class MoleculeCount:
    molecule_type: str
    count: int

@dataclass(frozen=True)
class MoleculeExclusionResult:
    molecule_counts: list[MoleculeCount]
    excluded_molecules: list[MoleculeCount]

@dataclass(frozen=True)
class AtomTemplate:
    local_nr: int
    atom_type: str
    residue_number: int
    residue_name: str
    atom_name: str
    charge: float


@dataclass
class MoleculeTemplate:
    molecule_type: str
    atoms: list[AtomTemplate] = field(default_factory=list)


@dataclass(frozen=True)
class ExpandedAtom:
    global_index: int
    global_nr: int
    residue_id: int
    molecule_type: str
    molecule_instance: int
    local_nr: int
    atom_type: str
    residue_number: int
    residue_name: str
    atom_name: str
    charge: float


def get_interaction_information(
    topology: Path,
    tpr: Path,
) -> InteractionInformation:
    topology = topology.expanduser().resolve()

    if not topology.exists():
        raise RedgewiseBuildError(f"topology file does not exist: {topology}")

    molecule_counts = parse_molecules_section(topology)
    needed_molecule_types = {item.molecule_type for item in molecule_counts}

    included_files = resolve_topology_includes(topology)

    atomtypes = parse_atomtypes_from_files(included_files)
    explicit_vdw = parse_nonbond_params_from_files(included_files)

    molecule_templates = parse_needed_molecule_templates(
        files=included_files,
        needed_molecule_types=needed_molecule_types,
    )

    validate_needed_templates(
        needed_molecule_types=needed_molecule_types,
        molecule_templates=molecule_templates,
    )

    expanded_atoms = expand_molecules(
    molecule_counts=molecule_counts,
    molecule_templates=molecule_templates,
    )

    tpr_atom_count = get_tpr_atom_count(tpr)

    if len(expanded_atoms) != tpr_atom_count:
        exclusion_result = try_excluding_trailing_molecules_to_match_tpr(
            molecule_counts=molecule_counts,
            molecule_templates=molecule_templates,
            topology_atom_count=len(expanded_atoms),
            tpr_atom_count=tpr_atom_count,
        )

        molecule_counts = exclusion_result.molecule_counts

        if exclusion_result.excluded_molecules:
            excluded = ", ".join(
                f"{item.molecule_type}({item.count})"
                for item in exclusion_result.excluded_molecules
            )
            print(
                "redgewise build: excluding trailing molecule entries from topology "
                f"to match TPR atom count. Excluded: {excluded}. "
                "Assuming these are solvent/ions or otherwise stripped trailing molecules."
            )

        expanded_atoms = expand_molecules(
            molecule_counts=molecule_counts,
            molecule_templates=molecule_templates,
        )

    validate_expanded_atoms_against_tpr(
        expanded_atoms=expanded_atoms,
        tpr=tpr,
    )

    info = build_interaction_information(
        expanded_atoms=expanded_atoms,
        atomtypes=atomtypes,
        explicit_vdw=explicit_vdw,
    )

    return info

def get_tpr_atom_count(tpr: Path) -> int:
    tpr = tpr.expanduser().resolve()

    if not tpr.exists():
        raise RedgewiseBuildError(f"TPR file does not exist: {tpr}")

    try:
        universe = mda.Universe(str(tpr))
    except Exception as exc:
        raise RedgewiseBuildError(f"could not load TPR with MDAnalysis: {tpr}") from exc

    return len(universe.atoms)


def try_excluding_trailing_molecules_to_match_tpr(
    molecule_counts: list[MoleculeCount],
    molecule_templates: dict[str, MoleculeTemplate],
    topology_atom_count: int,
    tpr_atom_count: int,
) -> MoleculeExclusionResult:
    if topology_atom_count < tpr_atom_count:
        raise RedgewiseBuildError(
            "expanded topology atom count is smaller than TPR atom count: "
            f"topology={topology_atom_count}, tpr={tpr_atom_count}. "
            "Cannot repair this by excluding trailing molecules."
        )

    if topology_atom_count == tpr_atom_count:
        return MoleculeExclusionResult(
            molecule_counts=molecule_counts,
            excluded_molecules=[],
        )

    kept = list(molecule_counts)
    excluded: list[MoleculeCount] = []

    while kept:
        excluded.insert(0, kept.pop())

        expanded_atoms = expand_molecules(
            molecule_counts=kept,
            molecule_templates=molecule_templates,
        )

        current_count = len(expanded_atoms)

        if current_count == tpr_atom_count:
            return MoleculeExclusionResult(
                molecule_counts=kept,
                excluded_molecules=excluded,
            )

        if current_count < tpr_atom_count:
            break

    excluded_text = ", ".join(
        f"{item.molecule_type}({item.count})"
        for item in excluded
    )

    raise RedgewiseBuildError(
        "expanded topology atom count does not match TPR atom count and could "
        "not be repaired by excluding trailing [ molecules ] entries:\n"
        f"  original topology atom count: {topology_atom_count}\n"
        f"  TPR atom count:               {tpr_atom_count}\n"
        f"  tried excluding:              {excluded_text}"
    )

def parse_molecules_section(topology: Path) -> list[MoleculeCount]:
    sections = parse_sections(topology)
    rows = sections.get("molecules", [])

    if not rows:
        raise RedgewiseBuildError(
            f"topology contains no [ molecules ] section: {topology}"
        )

    molecule_counts: list[MoleculeCount] = []

    for row in rows:
        fields = row.split()

        if len(fields) < 2:
            continue

        try:
            count = int(fields[1])
        except ValueError as exc:
            raise RedgewiseBuildError(
                f"cannot parse [ molecules ] line in {topology}: {row}"
            ) from exc

        molecule_counts.append(
            MoleculeCount(
                molecule_type=fields[0],
                count=count,
            )
        )

    if not molecule_counts:
        raise RedgewiseBuildError(
            f"no molecule entries found in [ molecules ] section: {topology}"
        )

    return molecule_counts


def resolve_topology_includes(topology: Path) -> list[Path]:
    topology = topology.expanduser().resolve()

    if not topology.exists():
        raise RedgewiseBuildError(f"topology file does not exist: {topology}")

    files: list[Path] = []
    seen: set[Path] = set()

    include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"')

    def visit(path: Path) -> None:
        path = path.expanduser().resolve()

        if path in seen:
            return

        if not path.exists():
            raise RedgewiseBuildError(f"included topology file does not exist: {path}")

        seen.add(path)
        files.append(path)

        base_dir = path.parent

        for raw_line in path.read_text().splitlines():
            line = raw_line.split(";", 1)[0].strip()
            match = include_pattern.match(line)

            if not match:
                continue

            include_path = Path(match.group(1)).expanduser()

            if not include_path.is_absolute():
                include_path = base_dir / include_path

            visit(include_path)

    visit(topology)

    return files


def parse_sections(path: Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.split(";", 1)[0].strip()

        if not line:
            continue

        section_match = re.match(r"^\[\s*([^\]]+?)\s*\]", line)

        if section_match:
            current_section = section_match.group(1).strip()
            sections.setdefault(current_section, [])
            continue

        if current_section is not None:
            sections[current_section].append(line)

    return sections

def parse_molecule_templates_from_file(path: Path) -> list[MoleculeTemplate]:
    templates: list[MoleculeTemplate] = []

    current_molecule_type: str | None = None
    current_atoms: list[AtomTemplate] = []
    current_section: str | None = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.split(";", 1)[0].strip()

        if not line:
            continue

        section_match = re.match(r"^\[\s*([^\]]+?)\s*\]", line)

        if section_match:
            new_section = section_match.group(1).strip()

            if new_section == "moleculetype":
                if current_molecule_type is not None:
                    templates.append(
                        MoleculeTemplate(
                            molecule_type=current_molecule_type,
                            atoms=current_atoms,
                        )
                    )

                current_molecule_type = None
                current_atoms = []

            current_section = new_section
            continue

        if current_section == "moleculetype":
            fields = line.split()

            if not fields:
                continue

            current_molecule_type = fields[0]
            continue

        if current_section == "atoms" and current_molecule_type is not None:
            fields = line.split()

            if len(fields) < 7:
                continue

            try:
                current_atoms.append(
                    AtomTemplate(
                        local_nr=int(fields[0]),
                        atom_type=fields[1],
                        residue_number=int(fields[2]),
                        residue_name=fields[3],
                        atom_name=fields[4],
                        charge=float(fields[6]),
                    )
                )
            except ValueError as exc:
                raise RedgewiseBuildError(
                    f"cannot parse [ atoms ] line in {path}: {line}"
                ) from exc

    if current_molecule_type is not None:
        templates.append(
            MoleculeTemplate(
                molecule_type=current_molecule_type,
                atoms=current_atoms,
            )
        )

    return [template for template in templates if template.atoms]

def parse_needed_molecule_templates(
    files: list[Path],
    needed_molecule_types: set[str],
) -> dict[str, MoleculeTemplate]:
    templates: dict[str, MoleculeTemplate] = {}

    for path in files:
        for template in parse_molecule_templates_from_file(path):
            if template.molecule_type not in needed_molecule_types:
                continue

            templates[template.molecule_type] = template

    return templates


def parse_moleculetype(
    sections: dict[str, list[str]],
    source: Path,
) -> str:
    rows = sections.get("moleculetype", [])

    if not rows:
        raise RedgewiseBuildError(f"missing [ moleculetype ] section in {source}")

    fields = rows[0].split()

    if not fields:
        raise RedgewiseBuildError(f"empty [ moleculetype ] section in {source}")

    return fields[0]


def parse_atom_templates(
    sections: dict[str, list[str]],
    source: Path,
) -> list[AtomTemplate]:
    atoms: list[AtomTemplate] = []

    for row in sections.get("atoms", []):
        fields = row.split()

        if len(fields) < 7:
            continue

        try:
            atoms.append(
                AtomTemplate(
                    local_nr=int(fields[0]),
                    atom_type=fields[1],
                    residue_number=int(fields[2]),
                    residue_name=fields[3],
                    atom_name=fields[4],
                    charge=float(fields[6]),
                )
            )
        except ValueError as exc:
            raise RedgewiseBuildError(
                f"cannot parse [ atoms ] line in {source}: {row}"
            ) from exc

    if not atoms:
        raise RedgewiseBuildError(f"no atoms found in [ atoms ] section of {source}")

    return atoms


def validate_needed_templates(
    needed_molecule_types: set[str],
    molecule_templates: dict[str, MoleculeTemplate],
) -> None:
    missing = sorted(needed_molecule_types - set(molecule_templates))

    if missing:
        raise RedgewiseBuildError(
            "topology [ molecules ] requests molecule types with no matching "
            "[ moleculetype ] definition in included files: "
            + ", ".join(missing)
        )


def expand_molecules(
    molecule_counts: list[MoleculeCount],
    molecule_templates: dict[str, MoleculeTemplate],
) -> list[ExpandedAtom]:
    expanded_atoms: list[ExpandedAtom] = []

    global_index = 0
    global_nr = 1
    global_residue_id = 1
    molecule_instance_by_type: dict[str, int] = {}

    for molecule_count in molecule_counts:
        molecule_type = molecule_count.molecule_type
        template = molecule_templates[molecule_type]

        for _ in range(molecule_count.count):
            molecule_instance_by_type[molecule_type] = (
                molecule_instance_by_type.get(molecule_type, 0) + 1
            )
            molecule_instance = molecule_instance_by_type[molecule_type]

            residue_map: dict[int, int] = {}

            for atom in template.atoms:
                if atom.residue_number not in residue_map:
                    residue_map[atom.residue_number] = global_residue_id
                    global_residue_id += 1

                expanded_atoms.append(
                    ExpandedAtom(
                        global_index=global_index,
                        global_nr=global_nr,
                        residue_id=residue_map[atom.residue_number],
                        molecule_type=molecule_type,
                        molecule_instance=molecule_instance,
                        local_nr=atom.local_nr,
                        atom_type=atom.atom_type,
                        residue_number=atom.residue_number,
                        residue_name=atom.residue_name,
                        atom_name=atom.atom_name,
                        charge=atom.charge,
                    )
                )

                global_index += 1
                global_nr += 1

    return expanded_atoms


def validate_expanded_atoms_against_tpr(
    expanded_atoms: list[ExpandedAtom],
    tpr: Path,
) -> None:
    tpr = tpr.expanduser().resolve()

    if not tpr.exists():
        raise RedgewiseBuildError(f"TPR file does not exist: {tpr}")

    try:
        universe = mda.Universe(str(tpr))
    except Exception as exc:
        raise RedgewiseBuildError(f"could not load TPR with MDAnalysis: {tpr}") from exc

    if len(expanded_atoms) != len(universe.atoms):
        raise RedgewiseBuildError(
            "expanded topology atom count does not match TPR atom count: "
            f"topology={len(expanded_atoms)}, tpr={len(universe.atoms)}"
        )

    for expanded_atom, tpr_atom in zip(expanded_atoms, universe.atoms, strict=True):
        mismatches: list[str] = []

        if expanded_atom.atom_name != tpr_atom.name:
            mismatches.append(
                f"name topology={expanded_atom.atom_name!r} tpr={tpr_atom.name!r}"
            )

        if expanded_atom.atom_type != tpr_atom.type:
            mismatches.append(
                f"type topology={expanded_atom.atom_type!r} tpr={tpr_atom.type!r}"
            )

        if expanded_atom.residue_name != tpr_atom.resname:
            mismatches.append(
                "residue "
                f"topology={expanded_atom.residue_name!r} "
                f"tpr={tpr_atom.resname!r}"
            )

        if mismatches:
            raise RedgewiseBuildError(
                "expanded topology does not match TPR at atom index "
                f"{expanded_atom.global_index} / atom nr {expanded_atom.global_nr}:\n"
                + "\n".join(f"  - {item}" for item in mismatches)
            )


def parse_atomtypes_from_files(files: list[Path]) -> dict[str, dict]:
    atomtypes: dict[str, dict] = {}

    for path in files:
        sections = parse_sections(path)

        for row in sections.get("atomtypes", []):
            fields = row.split()

            if len(fields) < 6:
                continue

            name = fields[0]

            try:
                float(fields[1])
                sigma = float(fields[4])
                epsilon = float(fields[5])
            except ValueError:
                if len(fields) < 7:
                    raise RedgewiseBuildError(
                        f"cannot parse [ atomtypes ] line in {path}: {row}"
                    )
                sigma = float(fields[5])
                epsilon = float(fields[6])

            atomtypes[name] = {
                "sigma": sigma,
                "epsilon": epsilon,
                "source": str(path),
            }

    return atomtypes


def parse_nonbond_params_from_files(files: list[Path]) -> dict[tuple[str, str], VdwInteraction]:
    interactions: dict[tuple[str, str], VdwInteraction] = {}

    for path in files:
        sections = parse_sections(path)

        for row in sections.get("nonbond_params", []):
            fields = row.split()

            if len(fields) < 5:
                continue

            type_i = fields[0]
            type_j = fields[1]

            try:
                sigma = float(fields[3])
                epsilon = float(fields[4])
            except ValueError as exc:
                raise RedgewiseBuildError(
                    f"cannot parse [ nonbond_params ] line in {path}: {row}"
                ) from exc

            key = pair_key(type_i, type_j)

            interactions[key] = VdwInteraction(
                type_i=key[0],
                type_j=key[1],
                sigma=sigma,
                epsilon=epsilon,
                source=f"[ nonbond_params ] in {path}",
            )

    return interactions


def build_interaction_information(
    expanded_atoms: list[ExpandedAtom],
    atomtypes: dict[str, dict],
    explicit_vdw: dict[tuple[str, str], VdwInteraction],
) -> InteractionInformation:
    info = InteractionInformation()

    for atom in expanded_atoms:
        info.add_atom_to_residue(
            residue_id=atom.residue_id,
            residue_name=atom.residue_name,
            molecule_type=atom.molecule_type,
            molecule_instance=atom.molecule_instance,
            nr=atom.global_nr,
            atom_type=atom.atom_type,
            atom_name=atom.atom_name,
            charge=atom.charge,
        )

    used_bead_types = sorted({atom.atom_type for atom in expanded_atoms})

    missing_atomtypes = sorted(set(used_bead_types) - set(atomtypes))
    if missing_atomtypes:
        raise RedgewiseBuildError(
            "bead types used by the expanded topology are missing from "
            "[ atomtypes ]: "
            + ", ".join(missing_atomtypes)
        )

    add_vdw_interactions(
        info=info,
        used_bead_types=used_bead_types,
        atomtypes=atomtypes,
        explicit_vdw=explicit_vdw,
    )

    return info


def add_vdw_interactions(
    info: InteractionInformation,
    used_bead_types: list[str],
    atomtypes: dict[str, dict],
    explicit_vdw: dict[tuple[str, str], VdwInteraction],
) -> None:
    for i, type_i in enumerate(used_bead_types):
        for type_j in used_bead_types[i:]:
            key = pair_key(type_i, type_j)

            if key in explicit_vdw:
                info.vdw_by_type_pair[key] = explicit_vdw[key]
                continue

            sigma_i = atomtypes[type_i]["sigma"]
            sigma_j = atomtypes[type_j]["sigma"]

            epsilon_i = atomtypes[type_i]["epsilon"]
            epsilon_j = atomtypes[type_j]["epsilon"]

            sigma = 0.5 * (sigma_i + sigma_j)
            epsilon = (epsilon_i * epsilon_j) ** 0.5

            info.add_vdw_interaction(
                type_i=type_i,
                type_j=type_j,
                sigma=sigma,
                epsilon=epsilon,
                source="mixed from [ atomtypes ] using Lorentz-Berthelot",
            )