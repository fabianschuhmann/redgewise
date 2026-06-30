# redgewise plot

Plot or export analyses from a redgewise build output.

## Subcommands

```bash
redgewise plot vmd --man
redgewise plot shortest_path --man
redgewise plot neighbors --man
```

## Plot types

- `vmd`: export pseudo-bead PDB and TCL drawer files for VMD.
- `shortest_path`: plot network-geodesic interaction distance profiles.
- `neighbors`: plot direct-neighbor edge summaries without shortest paths.

## Common concepts

Most plot commands operate on an analysis edge value:

```text
vdw, cl, vdw+cl, dvdw, dcl, dvdw+dcl
```

Most plot commands also support exclusions. Exclusions hide or remove vertices
depending on the command. See each subcommand manual for exact semantics.

## RAVE heatmaps

```bash
redgewise plot rave --man
```

Plot signed residue-by-frame direct-neighbor interactions between two or more disjoint regions. Each unordered region pair gets one subplot, and the two directional matrices are overlaid.
