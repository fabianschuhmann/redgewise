# redgewise build

Build a sparse, undirected residue/mixed-resolution interaction network from a
GROMACS topology, MDP file, structure, and trajectory.

## Synopsis

```bash
redgewise build \
  -s fixed.tpr \
  -t trajectory.xtc \
  -p topol.top \
  -f md.mdp \
  -o /tmp/output
```

## Required arguments

- `-s, --tpr PATH`: GROMACS TPR/topology structure file for MDAnalysis loading.
- `-t, --trajectory PATH`: trajectory or coordinate source.
- `-p, --topology PATH`: GROMACS topology file.
- `-f, --mdp PATH`: MDP file containing nonbonded cutoffs.
- `-o, --output PATH`: output directory.

## Output layout

```text
output/
  metadata.json
  vertices.parquet
  vertex_members.parquet
  edges.parquet
  values/
    part-00000.parquet
    part-00001.parquet
```

## Resolution overrides

Default behavior is one vertex per residue. Resolution overrides change this:

- `--high_res SELECTOR`: matched atoms become atom-level vertices.
- `--low_res SELECTOR`: matched atoms are merged into one low-resolution vertex.
- `--bundle SELECTOR [SELECTOR ...]`: listed selectors are merged into one bundle vertex.

Precedence:

```text
high_res > low_res > bundle > default residue
```

## Selector examples

```bash
--high_res ARG:76
--high_res "resname ARG and resid 76"
--high_res "resname ARG and resid 76-80"
--low_res W
--bundle POPC POPE CHOL
```

## Coordinate units

The compute backend normalizes coordinates to nm internally. It uses local
excluded/bonded topology distances as the primary coordinate-scale inference
source and writes the inferred scale to `metadata.json`.

## Performance options

- `--stride N`: analyze every Nth frame.
- `--frames-per-part N`: write this many processed frames per values parquet part.
- `--workers N`: requested worker count.
- `--gpu`: request GPU backend if implemented/available.

## Example

```bash
redgewise build \
  -s fixed.tpr \
  -t fixed.tpr \
  -p topol.top \
  -f md.mdp \
  -o /tmp/output \
  --stride 1000 \
  --frames-per-part 1 \
  --high_res ARG:76 \
  --bundle POPC POPE CHOL
```
