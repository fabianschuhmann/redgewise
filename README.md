# redgewise

**Residue Energy edge-wise analysis for molecular dynamics simulations.**

`redgewise` builds sparse, mixed-resolution nonbonded interaction-energy networks from GROMACS simulations and provides command-line tools for inspection, plotting, and downstream notebook analysis.

The main target workflow is:

1. compute residue-level or mixed-resolution interaction networks from a trajectory,
2. store the result in a compact sparse columnar format,
3. analyze edge values with consistent normalization and selector logic,
4. produce plots for VMD, shortest-path profiles, direct-neighbor profiles, and time-resolved region interaction heatmaps,
5. load networks and plot tables in Python/Jupyter through `redgewise.suite`.

---

## Features

- Build residue or mixed-resolution interaction-energy networks from GROMACS inputs.
- Supports VDW, Coulomb, combined energy, and radial derivative values.
- Sparse frame-wise edge-value storage using Parquet.
- Selector grammar for residues, atoms, molecule instances, residue names, atom names, and more.
- Mixed-resolution handling:
  - residue vertices,
  - atom-level high-resolution overrides,
  - coarse bundle vertices.
- Shared analysis layer for value selection, normalization, exclusions, and thresholds.
- Plotting tools:
  - VMD network export,
  - shortest-path residue profiles,
  - direct-neighbor residue profiles,
  - RAVE time-resolved region interaction heatmaps.
- Notebook interface via:

```python
from redgewise import suite
```

---

## Installation

From a local checkout:

```bash
git clone <repository-url>
cd redgewise
python -m pip install -e .
```

Dependencies are declared in `pyproject.toml` and include:

- `MDAnalysis`
- `numpy`
- `pyarrow`
- `tqdm`
- `matplotlib`
- `scipy`

---

## Command overview

```bash
redgewise --help
```

The command-line help is intentionally compact. Full manuals are available with `--man`:

```bash
redgewise --man
redgewise build --man
redgewise info --man
redgewise plot --man
redgewise plot vmd --man
redgewise plot shortest_path --man
redgewise plot neighbors --man
redgewise plot rave --man
```

---

## Building a network

Example:

```bash
redgewise build \
  -s output/md.tpr \
  -t output/md_processed.xtc \
  -p topol.top \
  -f prep/files/mdp/md.mdp \
  -o redgewise \
  --stride 10 \
  --frames-per-part 1
```

By default, each residue becomes one residue-level vertex.

Resolution overrides can be applied with selector-like syntax, for example:

```bash
redgewise build \
  -s fixed.tpr \
  -t traj.xtc \
  -p topol.top \
  -f md.mdp \
  -o redgewise \
  --high-res "ARG:76"
```

Resolution precedence is:

```text
high_res > low_res > bundle > default residue-level vertex
```

---

## Build output format

A build output directory has the following structure:

```text
redgewise/
├── metadata.json
├── vertices.parquet
├── vertex_members.parquet
├── edges.parquet
└── values/
    ├── part-00000.parquet
    ├── part-00001.parquet
    └── ...
```

### `vertices.parquet`

One row per network vertex.

Typical columns:

```text
vertex_id
label
kind
residue_name
residue_id
molecule_type
molecule_instance
atom_nr
atom_name
atom_type
charge
members
```

`kind` can include:

- `residue`
- `atom`
- `bundle`

### `vertex_members.parquet`

Maps vertices to atoms.

```text
vertex_id
atom_index
atom_nr
```

### `edges.parquet`

One row per edge observed at least once.

```text
edge_key
vertex1
vertex2
```

Edges are undirected. The key rule is:

```text
edge_key = vertex1 * n_vertices + vertex2 with vertex1 < vertex2
```

### `values/*.parquet`

Sparse frame-wise edge values.

```text
frame
edge_key
vdw
coulomb
vdw_dif
coulomb_dif
n_atom_pairs
```

Sparse semantics:

- Missing frame-edge rows mean zero.
- Edges not present in `edges.parquet` were never observed.
- No self-interactions are reported.
- Values are raw sums unless a later analysis or plotting step applies normalization.

---

## Energies and units

Internal compute units:

| Quantity | Unit |
|---|---|
| coordinates | nm |
| distances | nm |
| cutoffs | nm |
| VDW sigma | nm |
| energy | kJ/mol |
| derivative | kJ/mol/nm |

Implemented pair terms:

```text
V_LJ(r) = 4 ε [(σ/r)^12 - (σ/r)^6]
dV_LJ/dr = 24 ε / r [-(2(σ/r)^12) + (σ/r)^6]

V_coul(r) = 138.935458 q_i q_j / r
dV_coul/dr = -138.935458 q_i q_j / r²
```

The build step infers the numeric coordinate scale internally because MDAnalysis readers can expose different raw numeric units depending on the input source. The output metadata records the inferred scale.

---

## Inspecting a network

```bash
redgewise info -i redgewise
```

Show a longer listing:

```bash
redgewise info -i redgewise --list --long
```

Show selector-relevant fields and example selectors:

```bash
redgewise info -i redgewise --selector
```

---

## Selector syntax

Several commands accept selectors, including `info`, `plot shortest_path`, `plot neighbors`, `plot rave`, and resolution overrides during build.

Examples:

```text
molecule_instance 0
molinstance 0
molecule_instance in 1,2,3
resname ARG and resid 76
resid 1-100 and molecule_instance 0
kind atom and resname ARG
resname in POPC,POPE,POPS
not kind bundle
(resname ARG or resname LYS) and molecule_instance 0
```

Supported logical operators:

```text
and
or
not
(...)
```

Common fields and aliases:

| Field | Aliases |
|---|---|
| `vertex_id` | `id` |
| `residue_name` | `resname` |
| `residue_id` | `resid` |
| `molecule_type` | `moltype` |
| `molecule_instance` | `molinstance` |
| `atom_name` | `name` |
| `atom_type` | `type` |
| `atom_nr` | `bynum` |

---

## Value names

Most analysis and plotting commands accept:

| Name | Meaning |
|---|---|
| `vdw` | Lennard-Jones / VDW energy |
| `cl` | Coulomb energy |
| `coulomb` | alias for `cl` |
| `vdw+cl` | VDW plus Coulomb |
| `dvdw` | radial derivative of VDW energy |
| `dcl` | radial derivative of Coulomb energy |
| `dvdw+dcl` | combined radial derivative |

---

## Normalization modes

Shared analysis and plotting code supports:

| Mode | Meaning |
|---|---|
| `none` | raw edge sum |
| `per_atom_pair` | divide by contributing atom-pair count |
| `per_vertex_member_sqrt` | divide by `sqrt(n_members1 * n_members2)` |
| `per_vertex_member_product` | divide by `n_members1 * n_members2` |
| `per_coarse_member_sqrt` | apply sqrt member normalization only to coarse endpoints |
| `per_coarse_member_product` | apply product member normalization only to coarse endpoints |

---

## VMD network export

```bash
redgewise plot vmd \
  -i redgewise \
  -s fixed.tpr \
  -o redgewise_vmd \
  --value vdw+cl \
  --exclude-kind bundle
```

Outputs include:

```text
network_beads.pdb
network_beads.labels.tsv
drawer.tcl
load_network.tcl
edge_summary.parquet
```

Open `load_network.tcl` from VMD to inspect the network.

---

## Shortest-path profile

```bash
redgewise plot shortest_path \
  -i redgewise \
  -o shortest_path.png \
  --value vdw+cl \
  --exclude-kind bundle
```

The graph is built at the mixed-resolution vertex level and collapsed to a residue profile for plotting.

For energy values, edge distance is:

```text
1 / abs(edge_value)
```

For derivative values, edge distance is:

```text
abs(edge_value)
```

Targeted profile example:

```bash
redgewise plot shortest_path \
  -i redgewise \
  -o shortest_path_to_mol0.png \
  --value vdw+cl \
  --target "molecule_instance 0"
```

---

## Direct-neighbor profile

```bash
redgewise plot neighbors \
  -i redgewise \
  -o neighbors.png \
  --value vdw+cl
```

This plot does not use shortest paths. It summarizes direct incident edges for each source vertex and collapses the result to residues.

Targeted direct-neighbor profile:

```bash
redgewise plot neighbors \
  -i redgewise \
  -o neighbors_to_mol0.png \
  --value vdw+cl \
  --target "molecule_instance 0"
```

Summary modes include:

```text
mean
mean_abs
sum
sum_abs
median
median_abs
```

---

## RAVE plot

RAVE is a time-resolved direct-neighbor interaction heatmap.

```bash
redgewise plot rave \
  -i redgewise \
  -o rave.png \
  --value vdw+cl \
  --region "molecule_instance 1" --region-label A \
  --region "molecule_instance 2" --region-label B \
  --region "molecule_instance 3" --region-label C \
  --alpha .95 \
  --darkmode
```

RAVE semantics:

- x-axis: source residues;
- y-axis: frame;
- color: signed direct interaction value toward another selected region;
- regions must be disjoint;
- within-region interactions are ignored;
- all unordered region pairs are plotted;
- both directions are overlaid in each region-pair panel.

For three regions `A`, `B`, and `C`, panels are:

```text
A↔B
A↔C
B↔C
```

RAVE uses a fixed signed root color transform with exponent `0.4`. Colorbar tick labels remain in the original data units.

Outputs:

```text
rave.png
rave.tsv
rave_01.png
rave_02.png
...
```

---

## Python / Jupyter interface

The notebook interface is available as:

```python
from redgewise import suite
```

Load a build output:

```python
net = suite.load_network("redgewise")
net.summary()
```

Access core tables:

```python
vertices = net.vertices
edges = net.edges
members = net.vertex_members

vertices.head(5).to_dicts()
edges.head(5).to_dicts()
```

Select vertices:

```python
protein = net.select_vertices("molecule_instance 1")
lipids = net.select_vertices("resname in POPC,POPE,CHOL")
```

Define disjoint regions:

```python
regions = net.regions(
    [
        "molecule_instance 1",
        "molecule_instance 2",
        "molecule_instance 3",
    ],
    labels=["A", "B", "C"],
)
```

Load sparse frame-wise edge values:

```python
ev = net.edge_values(value="vdw+cl", normalization="none")
ev.head(10).to_dicts()
```

Filter values:

```python
frame0 = ev.where(ev["frame"] == 0)
strong = ev.where(abs(ev["value"]) > 10.0)
```

Load TSV files created by plotting commands:

```python
rave = suite.load_tsv("rave.tsv")
neighbors = suite.load_plot_table("neighbors.tsv")
```

Load multiple plot tables:

```python
tables = suite.load_plot_tables("*.tsv")
```

The suite API deliberately avoids pandas. Tables are lightweight NumPy-backed wrappers with methods such as:

```python
table["column"]
table.column("column")
table.head(10)
table.where(mask)
table.take(indices)
table.unique("column")
table.to_dicts(limit=10)
table.write_tsv("out.tsv")
```

---

## Development notes

This project is currently in active development. The file format and CLI are intended to be stable enough for internal analysis, but some details may still evolve.

Known areas for future refinement:

- verify topology parsing for sigma/epsilon versus C6/C12 parameter forms;
- improve VMD PBC minimum-image drawing for very long edges;
- expose plot-equivalent dense missing-zero semantics through the suite API;
- add more notebook helpers around RAVE matrices and neighbor summaries;
- harden documentation and examples against more simulation systems.

---

## License

Add license information here.
