# redgewise

Residue Energy edge-wise analysis from GROMACS simulations.

## Synopsis

```bash
redgewise build ...
redgewise info ...
redgewise plot vmd ...
redgewise plot shortest_path ...
redgewise plot neighbors ...
```

## Command manuals

```bash
redgewise build --man
redgewise info --man
redgewise plot --man
redgewise plot vmd --man
redgewise plot shortest_path --man
redgewise plot neighbors --man
```

## Typical workflow

1. Build a sparse interaction network from a GROMACS system and trajectory.
2. Inspect the output with `redgewise info`.
3. Plot direct-neighbor or shortest-path profiles.
4. Export VMD files for structural inspection.

## Notes

All computed nonbonded values are stored in GROMACS-compatible units:
coordinates in nm, energies in kJ/mol, and derivatives in kJ/mol/nm.
MDAnalysis coordinate-unit quirks are handled in the build compute layer.
