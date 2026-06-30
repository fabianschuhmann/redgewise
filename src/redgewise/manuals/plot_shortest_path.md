# redgewise plot shortest_path

Plot source-wise mean shortest-path distances through the retained network.

## Synopsis

```bash
redgewise plot shortest_path \
  -i /tmp/output \
  -o /tmp/shortest_path.png \
  --value vdw+cl
```

## Core semantics

The graph is built at the actual vertex resolution. Results are then summarized
into a residue-collapsed profile and, when atom-level vertices exist, a high-
resolution detail profile.

For energy-like values, stronger interactions are normally shorter graph edges.
For derivative values, recent builds may use absolute derivative as the path
cost depending on the implemented path-weight logic.

## Target selector

```bash
--target "molecule_instance 0"
```

Without `--target`, each source vertex is averaged to all reachable retained
vertices. With `--target`, the shortest paths may still traverse the whole
retained graph, but the final average only includes selected target vertices.
If the source is itself in the target set, the self-distance is excluded.

## Exclusions

```bash
--exclude-kind bundle
--exclude-resname POPC
--exclude-vertex-id 299
```

Excluded vertices are removed before graph construction.

## Display options

- `--molecule-delimiter-min-size N`: draw molecule-boundary delimiters when the
  left or right molecule block contains at least N plotted residue points.
- `--renumber-molecule-residues`: label residue ticks as 1..N within molecule
  blocks while keeping original residue IDs in the TSV.

## Selector examples

```bash
--target "molecule_instance 0"
--target "molecule_instance in 1,2,3"
--target "resname ARG and resid 76"
--target "resid 1-100 and molecule_instance 0"
--target "not kind bundle"
```

## Examples

```bash
redgewise plot shortest_path \
  -i /tmp/output \
  -o /tmp/to_mol0.png \
  --value vdw+cl \
  --target "molecule_instance 0" \
  --molecule-delimiter-min-size 5
```

```bash
redgewise plot shortest_path \
  -i /tmp/output \
  -o /tmp/no_bundle.png \
  --exclude-kind bundle
```
