# redgewise plot vmd

Export a VMD visualization package from a redgewise output directory.

## Synopsis

```bash
redgewise plot vmd \
  -i /tmp/output \
  -s fixed.tpr \
  -o /tmp/output_vmd \
  --value vdw+cl
```

## Outputs

```text
output_vmd/
  network_beads.pdb
  network_beads.labels.tsv
  drawer.tcl
  load_network.tcl
  edge_summary.parquet
```

## Values

- `vdw`: VDW/LJ energy.
- `cl`: Coulomb energy.
- `vdw+cl`: combined energy.
- `dvdw`: VDW radial derivative.
- `dcl`: Coulomb radial derivative.
- `dvdw+dcl`: combined derivative.

## Normalization

- `none`: raw mean edge values.
- `per_atom_pair`: divide by atom-pair count.
- `per_vertex_member_sqrt`: divide by sqrt of endpoint member counts.
- `per_vertex_member_product`: divide by endpoint member-count product.
- `per_coarse_member_sqrt`: apply sqrt member normalization only to coarse endpoints.
- `per_coarse_member_product`: apply product member normalization only to coarse endpoints.

## Radius modes

- `linear`: radius = scale * abs(value)
- `sqrt`: radius = scale * sqrt(abs(value))
- `log`: radius = scale * log1p(abs(value))

## Exclusions

```bash
--exclude-kind bundle
--exclude-resname POPC
--exclude-label ARG:76
--exclude-vertex-id 299
```

Excluded vertices are not drawn as VMD pseudo-beads/edges.

## Coordinate unit

The VMD PDB/TCL output is written in Angstrom. Use `--coordinate-unit nm` if the
structure reader returns nm-scale coordinates, such as some TPR reader paths.

## Examples

```bash
redgewise plot vmd \
  -i /tmp/output \
  -s fixed.tpr \
  -o /tmp/vmd \
  --coordinate-unit nm \
  --normalize per_coarse_member_product \
  --radius-mode log \
  --radius-scale 0.05
```

```bash
redgewise plot vmd \
  -i /tmp/output \
  -s fixed.tpr \
  -o /tmp/vmd_dvdw \
  --value dvdw \
  --exclude-kind bundle
```
