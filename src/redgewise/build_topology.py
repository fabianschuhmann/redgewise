from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import MDAnalysis as mda

from redgewise.build_information import InteractionInformation, pair_key


class RedgewiseBuildError(Exception):
    """Expected topology/build error with a user-readable message."""


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
    nrexcl: int
    atoms: list[AtomTemplate] = field(default_factory=list)
    bonds: list[tuple[int, int]] = field(default_factory=list)
    constraints: list[tuple[int, int]] = field(default_factory=list)
    exclusions: list[tuple[int, int]] = field(default_factory=list)
    pairs: list[tuple[int, int]] = field(default_factory=list)


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


@dataclass(frozen=True)
class AtomTypeParameters:
    sigma: float
    epsilon: float


def get_interaction_information(topology: Path, tpr: Path) -> InteractionInformation:
    molecule_counts = parse_molecules_section(topology)
    files = resolve_topology_includes(topology)

    needed_molecule_types = {entry.molecule_type for entry in molecule_counts}
    templates = parse_needed_molecule_templates(
        files=files,
        needed_molecule_types=needed_molecule_types,
    )

    validate_needed_templates(
        templates=templates,
        needed_molecule_types=needed_molecule_types,
    )

    tpr_atom_count = get_tpr_atom_count(tpr)

    exclusion_result = try_excluding_trailing_molecules_to_match_tpr(
        molecule_counts=molecule_counts,
        templates=templates,
        tpr_atom_count=tpr_atom_count,
    )

    molecule_counts = exclusion_result.molecule_counts

    if exclusion_result.excluded_molecules:
        excluded = ", ".join(
            f"{entry.molecule_type}({entry.count})"
            for entry in exclusion_result.excluded_molecules
        )
        print(
            "redgewise build: excluding trailing molecule entries from topology "
            "to match TPR atom count. "
            f"Excluded: {excluded}. "
            "Assuming these are solvent/ions or otherwise stripped trailing molecules."
        )

    expanded_atoms, excluded_atom_pairs, n_pair_entries = expand_molecules(
        molecule_counts=molecule_counts,
        templates=templates,
    )

    if n_pair_entries:
        print(
            "redgewise build: warning: [ pairs ] entries were detected. "
            "Special pair interactions are not implemented yet; these pairs are "
            "excluded from normal nonbonded interactions."
        )

    validate_expanded_atoms_against_tpr(
        expanded_atoms=expanded_atoms,
        tpr=tpr,
    )

    atomtypes = parse_atomtypes_from_files(files)
    nonbond_params = parse_nonbond_params_from_files(files)

    interaction_information = build_interaction_information(
        expanded_atoms=expanded_atoms,
        excluded_atom_pairs=excluded_atom_pairs,
    )

    add_vdw_interactions(
        interaction_information=interaction_information,
        atomtypes=atomtypes,
        nonbond_params=nonbond_params,
    )

    return interaction_information


def parse_molecules_section(topology: Path) -> list[MoleculeCount]:
    topology = topology.expanduser().resolve()

    if not topology.exists():
        raise RedgewiseBuildError(f"topology file does not exist: {topology}")

    sections = parse_sections(topology)
    rows = sections.get("molecules", [])

    if not rows:
        raise RedgewiseBuildError(f"topology has no [ molecules ] section: {topology}")

    molecule_counts: list[MoleculeCount] = []

    for row in rows:
        fields = row.split()
        if len(fields) < 2:
            continue

        try:
            molecule_counts.append(
                MoleculeCount(
                    molecule_type=fields[0],
                    count=int(fields[1]),
                )
            )
        except ValueError as exc:
            raise RedgewiseBuildError(
                f"cannot parse [ molecules ] line in {topology}: {row}"
            ) from exc

    if not molecule_counts:
        raise RedgewiseBuildError(
            f"topology has empty [ molecules ] section: {topology}"
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


def parse_molecule_templates_from_file(path: Path) -> list[MoleculeTemplate]:
    templates: list[MoleculeTemplate] = []

    current_template: MoleculeTemplate | None = None
    current_section: str | None = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.split(";", 1)[0].strip()

        if not line:
            continue

        section_match = re.match(r"^\[\s*([^\]]+?)\s*\]", line)

        if section_match:
            current_section = section_match.group(1).strip()

            if current_section == "moleculetype":
                if current_template is not None and current_template.atoms:
                    templates.append(current_template)
                current_template = None

            continue

        if current_section == "moleculetype":
            fields = line.split()
            if len(fields) < 2:
                continue

            try:
                current_template = MoleculeTemplate(
                    molecule_type=fields[0],
                    nrexcl=int(fields[1]),
                )
            except ValueError as exc:
                raise RedgewiseBuildError(
                    f"cannot parse [ moleculetype ] line in {path}: {line}"
                ) from exc

            continue

        if current_template is None:
            continue

        if current_section == "atoms":
            atom = parse_atom_template_line(path, line)
            if atom is not None:
                current_template.atoms.append(atom)

        elif current_section == "bonds":
            pair = parse_local_pair_line(line)
            if pair is not None:
                current_template.bonds.append(pair)

        elif current_section == "constraints":
            pair = parse_local_pair_line(line)
            if pair is not None:
                current_template.constraints.append(pair)

        elif current_section == "pairs":
            pair = parse_local_pair_line(line)
            if pair is not None:
                current_template.pairs.append(pair)

        elif current_section == "exclusions":
            current_template.exclusions.extend(parse_exclusion_line(line))

    if current_template is not None and current_template.atoms:
        templates.append(current_template)

    return templates


def parse_atom_template_line(path: Path, line: str) -> AtomTemplate | None:
    fields = line.split()

    if len(fields) < 7:
        return None

    try:
        return AtomTemplate(
            local_nr=int(fields[0]),
            atom_type=fields[1],
            residue_number=int(fields[2]),
            residue_name=fields[3],
            atom_name=fields[4],
            charge=float(fields[6]),
        )
    except ValueError as exc:
        raise RedgewiseBuildError(
            f"cannot parse [ atoms ] line in {path}: {line}"
        ) from exc


def parse_local_pair_line(line: str) -> tuple[int, int] | None:
    fields = line.split()

    if len(fields) < 2:
        return None

    try:
        return normalize_local_pair(int(fields[0]), int(fields[1]))
    except ValueError:
        return None


def parse_exclusion_line(line: str) -> list[tuple[int, int]]:
    fields = line.split()

    if len(fields) < 2:
        return []

    try:
        atom_i = int(fields[0])
        excluded_atoms = [int(field) for field in fields[1:]]
    except ValueError:
        return []

    return [
        normalize_local_pair(atom_i, atom_j)
        for atom_j in excluded_atoms
        if atom_i != atom_j
    ]


def validate_needed_templates(
    templates: dict[str, MoleculeTemplate],
    needed_molecule_types: set[str],
) -> None:
    missing = sorted(needed_molecule_types - set(templates))

    if missing:
        raise RedgewiseBuildError(
            "missing molecule template(s): " + ", ".join(missing)
        )


def expand_molecules(
    molecule_counts: list[MoleculeCount],
    templates: dict[str, MoleculeTemplate],
) -> tuple[list[ExpandedAtom], set[tuple[int, int]], int]:
    expanded_atoms: list[ExpandedAtom] = []
    excluded_atom_pairs: set[tuple[int, int]] = set()

    global_atom_nr = 1
    global_residue_id = 0
    global_molecule_instance = 0
    n_pair_entries = 0

    local_exclusions_by_type = {
        molecule_type: build_local_excluded_pairs(template)
        for molecule_type, template in templates.items()
    }

    for molecule_count in molecule_counts:
        template = templates[molecule_count.molecule_type]
        local_exclusions = local_exclusions_by_type[molecule_count.molecule_type]
        n_pair_entries += len(template.pairs) * molecule_count.count

        for _ in range(molecule_count.count):
            global_molecule_instance += 1
            molecule_first_global_atom_nr = global_atom_nr
            residue_id_by_local_residue: dict[int, int] = {}

            for atom in template.atoms:
                if atom.residue_number not in residue_id_by_local_residue:
                    global_residue_id += 1
                    residue_id_by_local_residue[atom.residue_number] = global_residue_id

                expanded_atoms.append(
                    ExpandedAtom(
                        global_index=global_atom_nr - 1,
                        global_nr=global_atom_nr,
                        residue_id=residue_id_by_local_residue[atom.residue_number],
                        molecule_type=template.molecule_type,
                        molecule_instance=global_molecule_instance,
                        local_nr=atom.local_nr,
                        atom_type=atom.atom_type,
                        residue_number=atom.residue_number,
                        residue_name=atom.residue_name,
                        atom_name=atom.atom_name,
                        charge=atom.charge,
                    )
                )

                global_atom_nr += 1

            for local_i, local_j in local_exclusions:
                global_i = molecule_first_global_atom_nr + local_i - 2
                global_j = molecule_first_global_atom_nr + local_j - 2

                excluded_atom_pairs.add(normalize_atom_index_pair(global_i, global_j))

    return expanded_atoms, excluded_atom_pairs, n_pair_entries


def build_local_excluded_pairs(template: MoleculeTemplate) -> set[tuple[int, int]]:
    excluded: set[tuple[int, int]] = set()

    local_atom_numbers = [atom.local_nr for atom in template.atoms]
    local_atom_set = set(local_atom_numbers)

    graph: dict[int, set[int]] = {local_nr: set() for local_nr in local_atom_numbers}

    for atom_i, atom_j in template.bonds + template.constraints:
        if atom_i not in local_atom_set or atom_j not in local_atom_set:
            continue

        graph[atom_i].add(atom_j)
        graph[atom_j].add(atom_i)

    if template.nrexcl > 0:
        for atom_i in local_atom_numbers:
            for atom_j in atoms_within_graph_depth(
                graph=graph,
                start=atom_i,
                max_depth=template.nrexcl,
            ):
                if atom_i == atom_j:
                    continue
                excluded.add(normalize_local_pair(atom_i, atom_j))

    for atom_i, atom_j in template.exclusions:
        if atom_i in local_atom_set and atom_j in local_atom_set:
            excluded.add(normalize_local_pair(atom_i, atom_j))

    for atom_i, atom_j in template.pairs:
        if atom_i in local_atom_set and atom_j in local_atom_set:
            excluded.add(normalize_local_pair(atom_i, atom_j))

    return excluded


def atoms_within_graph_depth(
    graph: dict[int, set[int]],
    start: int,
    max_depth: int,
) -> set[int]:
    seen = {start}
    reached: set[int] = set()
    queue: deque[tuple[int, int]] = deque([(start, 0)])

    while queue:
        atom, depth = queue.popleft()

        if depth >= max_depth:
            continue

        for neighbor in graph.get(atom, set()):
            if neighbor in seen:
                continue

            seen.add(neighbor)
            reached.add(neighbor)
            queue.append((neighbor, depth + 1))

    return reached


def get_tpr_atom_count(tpr: Path) -> int:
    tpr = tpr.expanduser().resolve()

    if not tpr.exists():
        raise RedgewiseBuildError(f"TPR file does not exist: {tpr}")

    universe = mda.Universe(str(tpr))
    return len(universe.atoms)


def try_excluding_trailing_molecules_to_match_tpr(
    molecule_counts: list[MoleculeCount],
    templates: dict[str, MoleculeTemplate],
    tpr_atom_count: int,
) -> MoleculeExclusionResult:
    topology_atom_count = count_atoms_from_molecule_counts(
        molecule_counts=molecule_counts,
        templates=templates,
    )

    if topology_atom_count == tpr_atom_count:
        return MoleculeExclusionResult(
            molecule_counts=molecule_counts,
            excluded_molecules=[],
        )

    if topology_atom_count < tpr_atom_count:
        raise RedgewiseBuildError(
            "expanded topology has fewer atoms than TPR: "
            f"topology={topology_atom_count}, tpr={tpr_atom_count}"
        )

    remaining = list(molecule_counts)
    excluded: list[MoleculeCount] = []

    while remaining:
        current_count = count_atoms_from_molecule_counts(
            molecule_counts=remaining,
            templates=templates,
        )

        if current_count == tpr_atom_count:
            return MoleculeExclusionResult(
                molecule_counts=remaining,
                excluded_molecules=excluded,
            )

        last = remaining.pop()
        excluded.append(last)

    raise RedgewiseBuildError(
        "could not match topology atom count to TPR by excluding trailing "
        "[ molecules ] entries"
    )


def count_atoms_from_molecule_counts(
    molecule_counts: list[MoleculeCount],
    templates: dict[str, MoleculeTemplate],
) -> int:
    total = 0

    for entry in molecule_counts:
        total += len(templates[entry.molecule_type].atoms) * entry.count

    return total


def validate_expanded_atoms_against_tpr(
    expanded_atoms: list[ExpandedAtom],
    tpr: Path,
) -> None:
    universe = mda.Universe(str(tpr))

    if len(universe.atoms) != len(expanded_atoms):
        raise RedgewiseBuildError(
            "expanded topology atom count does not match TPR atom count: "
            f"topology={len(expanded_atoms)}, tpr={len(universe.atoms)}"
        )

    for expanded_atom, tpr_atom in zip(expanded_atoms, universe.atoms):
        if expanded_atom.atom_name != tpr_atom.name:
            raise RedgewiseBuildError(
                "expanded topology atom order does not match TPR: "
                f"atom {expanded_atom.global_nr}: "
                f"topology={expanded_atom.atom_name}, tpr={tpr_atom.name}"
            )


def parse_atomtypes_from_files(files: list[Path]) -> dict[str, AtomTypeParameters]:
    atomtypes: dict[str, AtomTypeParameters] = {}

    for path in files:
        sections = parse_sections(path)

        for line in sections.get("atomtypes", []):
            fields = line.split()

            if len(fields) < 6:
                continue

            atom_type = fields[0]

            try:
                sigma = float(fields[-2])
                epsilon = float(fields[-1])
            except ValueError:
                continue

            atomtypes[atom_type] = AtomTypeParameters(
                sigma=sigma,
                epsilon=epsilon,
            )

    return atomtypes


def parse_nonbond_params_from_files(
    files: list[Path],
) -> dict[tuple[str, str], AtomTypeParameters]:
    nonbond_params: dict[tuple[str, str], AtomTypeParameters] = {}

    for path in files:
        sections = parse_sections(path)

        for line in sections.get("nonbond_params", []):
            fields = line.split()

            if len(fields) < 5:
                continue

            type_i = fields[0]
            type_j = fields[1]

            try:
                sigma = float(fields[-2])
                epsilon = float(fields[-1])
            except ValueError:
                continue

            nonbond_params[pair_key(type_i, type_j)] = AtomTypeParameters(
                sigma=sigma,
                epsilon=epsilon,
            )

    return nonbond_params


def build_interaction_information(
    expanded_atoms: list[ExpandedAtom],
    excluded_atom_pairs: set[tuple[int, int]],
) -> InteractionInformation:
    interaction_information = InteractionInformation()
    interaction_information.excluded_atom_pairs.update(excluded_atom_pairs)

    for atom in expanded_atoms:
        interaction_information.add_atom_to_residue(
            residue_id=atom.residue_id,
            residue_name=atom.residue_name,
            molecule_type=atom.molecule_type,
            molecule_instance=atom.molecule_instance,
            nr=atom.global_nr,
            atom_type=atom.atom_type,
            atom_name=atom.atom_name,
            charge=atom.charge,
        )

    return interaction_information


def add_vdw_interactions(
    interaction_information: InteractionInformation,
    atomtypes: dict[str, AtomTypeParameters],
    nonbond_params: dict[tuple[str, str], AtomTypeParameters],
) -> None:
    used_atom_types = {
        atom.atom_type
        for residue in interaction_information.residues.values()
        for atom in residue.atoms
    }

    missing = sorted(used_atom_types - set(atomtypes))

    if missing:
        raise RedgewiseBuildError(
            "missing [ atomtypes ] parameters for atom type(s): "
            + ", ".join(missing)
        )

    for type_i in sorted(used_atom_types):
        for type_j in sorted(used_atom_types):
            key = pair_key(type_i, type_j)

            if key in interaction_information.vdw_by_type_pair:
                continue

            if key in nonbond_params:
                parameters = nonbond_params[key]
                source = "nonbond_params"
            else:
                parameters_i = atomtypes[type_i]
                parameters_j = atomtypes[type_j]

                parameters = AtomTypeParameters(
                    sigma=0.5 * (parameters_i.sigma + parameters_j.sigma),
                    epsilon=math.sqrt(parameters_i.epsilon * parameters_j.epsilon),
                )
                source = "combination_rule_2"

            interaction_information.add_vdw_interaction(
                type_i=type_i,
                type_j=type_j,
                sigma=parameters.sigma,
                epsilon=parameters.epsilon,
                source=source,
            )


def normalize_local_pair(atom_i: int, atom_j: int) -> tuple[int, int]:
    if atom_i <= atom_j:
        return atom_i, atom_j
    return atom_j, atom_i


def normalize_atom_index_pair(atom_i: int, atom_j: int) -> tuple[int, int]:
    if atom_i == atom_j:
        raise RedgewiseBuildError("cannot exclude atom pair with identical atom index")

    if atom_i <= atom_j:
        return atom_i, atom_j
    return atom_j, atom_i