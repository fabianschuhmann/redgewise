# redgewise plot neighbors

Plot direct-neighbor edge summaries. This command does not compute shortest
paths.

## Synopsis

```bash
redgewise plot neighbors \
  -i /tmp/output \
  -o /tmp/neighbors.png \
  --value vdw+cl
```

## Core semantics

For every plotted source vertex:

1. Collect direct incident graph edges.
2. Optionally restrict neighbor endpoints with `--target SELECTOR`.
3. Summarize selected edge values with `--neighbor-summary`.
4. Collapse source-vertex summaries to residue-level rows for plotting.

## Neighbor summary modes

- `mean`: mean signed edge value.
- `mean_abs`: mean absolute edge value. Default.
- `sum`: summed signed edge value.
- `sum_abs`: summed absolute edge value.
- `median`: median signed edge value.
- `median_abs`: median absolute edge value.

## Exclude vs remove

Exclude hides source vertices from plotting but keeps them as possible endpoints:

```bash
--exclude-resname POPC
--exclude-kind bundle
```

Remove deletes vertices from all neighbor calculations:

```bash
--remove-resname POPC
--remove-kind bundle
```

## Split excluded endpoints

```bash
--split-excluded
```

When enabled, each plotted residue can receive separate categories:

- `non_excluded`: direct edges toward non-excluded endpoints.
- `excluded`: direct edges toward excluded endpoints.

## Target selector

```bash
--target "molecule_instance 0"
```

Only direct neighbor endpoints matching the selector are included in the source
summary. The selector does not make paths; this is still direct-neighbor only.

## Display options

- `--molecule-delimiter-min-size N`: draw molecule-boundary delimiters when the
  left or right molecule block contains at least N plotted residue points.
- `--renumber-molecule-residues`: label residue ticks as 1..N within molecule
  blocks while keeping original residue IDs in the TSV.

## Examples

```bash
redgewise plot neighbors \
  -i /tmp/output \
  -o /tmp/neighbors.png \
  --value dvdw \
  --neighbor-summary mean_abs
```

```bash
redgewise plot neighbors \
  -i /tmp/output \
  -o /tmp/neighbors_split.png \
  --exclude-resname POPC \
  --exclude-resname POPE \
  --split-excluded
```

```bash
redgewise plot neighbors \
  -i /tmp/output \
  -o /tmp/neighbors_to_mol0.png \
  --target "molecule_instance 0"
```
