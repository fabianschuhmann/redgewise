from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InteractionInformationSummary:
    n_residues: int
    n_atoms: int
    n_vdw_type_pairs: int
    residue_names: list[str]
    bead_types: list[str]

@dataclass(frozen=True)
class AtomInformation:
    nr: int
    atom_type: str
    atom_name: str
    charge: float


@dataclass
class ResidueInformation:
    residue_id: int
    residue_name: str
    molecule_type: str
    molecule_instance: int
    atoms: list[AtomInformation] = field(default_factory=list)

    def add_atom(
        self,
        nr: int,
        atom_type: str,
        atom_name: str,
        charge: float,
    ) -> None:
        self.atoms.append(
            AtomInformation(
                nr=nr,
                atom_type=atom_type,
                atom_name=atom_name,
                charge=charge,
            )
        )


@dataclass(frozen=True)
class VdwInteraction:
    type_i: str
    type_j: str
    sigma: float
    epsilon: float
    source: str


@dataclass
class InteractionInformation:
    residues: dict[int, ResidueInformation] = field(default_factory=dict)
    vdw_by_type_pair: dict[tuple[str, str], VdwInteraction] = field(
        default_factory=dict
    )
    excluded_atom_pairs: set[tuple[int, int]] = field(default_factory=set)

    def add_atom_to_residue(
        self,
        residue_id: int,
        residue_name: str,
        molecule_type: str,
        molecule_instance: int,
        nr: int,
        atom_type: str,
        atom_name: str,
        charge: float,
    ) -> None:
        if residue_id not in self.residues:
            self.residues[residue_id] = ResidueInformation(
                residue_id=residue_id,
                residue_name=residue_name,
                molecule_type=molecule_type,
                molecule_instance=molecule_instance,
            )

        self.residues[residue_id].add_atom(
            nr=nr,
            atom_type=atom_type,
            atom_name=atom_name,
            charge=charge,
        )

    def add_vdw_interaction(
        self,
        type_i: str,
        type_j: str,
        sigma: float,
        epsilon: float,
        source: str,
    ) -> None:
        key = pair_key(type_i, type_j)

        self.vdw_by_type_pair[key] = VdwInteraction(
            type_i=key[0],
            type_j=key[1],
            sigma=sigma,
            epsilon=epsilon,
            source=source,
        )

    def get_vdw_interaction(
        self,
        type_i: str,
        type_j: str,
    ) -> VdwInteraction:
        return self.vdw_by_type_pair[pair_key(type_i, type_j)]
    
    def summary(self) -> InteractionInformationSummary:
        residue_names = sorted(
            {residue.residue_name for residue in self.residues.values()}
        )

        bead_types = sorted(
            {
                atom.atom_type
                for residue in self.residues.values()
                for atom in residue.atoms
            }
        )

        n_atoms = sum(
            len(residue.atoms)
            for residue in self.residues.values()
        )

        return InteractionInformationSummary(
            n_residues=len(self.residues),
            n_atoms=n_atoms,
            n_vdw_type_pairs=len(self.vdw_by_type_pair),
            residue_names=residue_names,
            bead_types=bead_types,
        )


def pair_key(type_i: str, type_j: str) -> tuple[str, str]:
    return tuple(sorted((type_i, type_j)))