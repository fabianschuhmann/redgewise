# redgewise plot rave

Plot a time-resolved residue interaction heatmap between two or more disjoint regions.

RAVE is pair-based and direct-neighbor based. It does not compute graph shortest paths. For each unordered pair of selected regions, the plot overlays the two directional residue-by-frame matrices in one subplot:

- `A→B`: residues in region `A` interacting with all vertices in region `B`.
- `B→A`: residues in region `B` interacting with all vertices in region `A`.

The underlying network edges are undirected, but residue attribution is directional: the value assigned to a residue in `A` is the sum of its direct edges to region `B`, while the value assigned to a residue in `B` is the sum of its direct edges to region `A`.

## Usage

```bash
redgewise plot rave \
  -i /tmp/output \
  -o /tmp/rave.png \
  --value vdw+cl \
  --region "molecule_instance 0" \
  --region "molecule_instance 1"
```

With labels:

```bash
redgewise plot rave \
  -i /tmp/output \
  -o /tmp/rave.png \
  --value vdw+cl \
  --region "molecule_instance 0" --region-label A \
  --region "molecule_instance 1" --region-label B \
  --region "molecule_instance 2" --region-label C
```

## Required arguments

```text
-i, --input PATH
  Redgewise build output directory.

-o, --output PATH
  Output image path. If a directory is given, rave.png is written inside it.

--region SELECTOR
  Select one region. Repeat at least twice.
```

Region selectors use the shared redgewise selector grammar.

Examples:

```bash
--region "molecule_instance 0"
--region "molecule_type Protein and molecule_instance 0"
--region "resname POPC"
--region "resid 1-100 and molecule_instance 0"
```

## Region rules

Regions must be disjoint.

If any vertex matches more than one `--region`, the command exits with an error. This is deliberate: RAVE is designed for studying interactions and clustering between separate regions.

Only selected regions are considered. There are no `--exclude-*` or `--remove-*` flags for this plot. Edges are included only when both endpoints are in selected regions and the endpoints belong to different regions.

Within-region interactions are ignored:

```text
A-A ignored
B-B ignored
A-B included
```

## Value semantics

For each frame and each source residue, RAVE computes:

```text
sum of signed direct edge values from that residue to the target region
```

The values are not converted to absolute values. Negative and positive values remain distinguishable.

Supported values:

```text
vdw
cl
vdw+cl
dvdw
dcl
dvdw+dcl
```

Default:

```text
--value vdw+cl
```

## Normalization

The plot accepts the same normalization names as the profile plots:

```text
none
per_atom_pair
per_vertex_member_sqrt
per_vertex_member_product
per_coarse_member_sqrt
per_coarse_member_product
```

Default:

```text
--normalize none
```

For RAVE, normalization is applied at the per-frame edge-row level before residue/frame accumulation.

## Plot layout

Each unordered region pair is plotted as one subplot.

For three regions `A`, `B`, and `C`, the subplot panels are:

```text
A↔B
A↔C
B↔C
```

For each pair panel, both directional matrices are overlaid:

```text
A→B and B→A
```

At most three pair panels are placed in one row. Additional panels wrap to the next row.

## Colors and scale

The scale is signed and linear.

All subplots use one shared global magnitude scale:

```text
-vmax ... 0 ... +vmax
```

The two directions are drawn with different colorblind-conscious colors and a transparent alpha layer. Positive and negative values use different colors for each direction.

Default alpha:

```text
--alpha 0.7
```

## Dark mode

```bash
--darkmode
```

Only the inside of each subplot is black. The figure background, axes, ticks, labels, and surrounding whitespace remain in the normal style.

## Outputs

If the output is:

```bash
-o /tmp/rave.png
```

then the command writes:

```text
/tmp/rave.png
/tmp/rave.tsv
```

The TSV contains one row per nonzero residue/frame/direction entry and includes:

```text
frame
source_region
target_region
source_selector
target_selector
molecule_type
molecule_instance
residue_id
residue_name
residue_index
value
n_edges
value_name
normalization
```

## Examples

Protein chain interaction RAVE:

```bash
redgewise plot rave \
  -i /tmp/output \
  -o /tmp/rave_chains.png \
  --value vdw+cl \
  --region "molecule_instance 0" --region-label chain0 \
  --region "molecule_instance 1" --region-label chain1
```

Derivative interaction RAVE:

```bash
redgewise plot rave \
  -i /tmp/output \
  -o /tmp/rave_dvdw.png \
  --value dvdw \
  --region "molecule_instance 0" --region-label A \
  --region "molecule_instance 1" --region-label B \
  --darkmode
```

Three-region clustering plot:

```bash
redgewise plot rave \
  -i /tmp/output \
  -o /tmp/rave_three_regions.png \
  --value vdw+cl \
  --region "molecule_instance 0" --region-label A \
  --region "molecule_instance 1" --region-label B \
  --region "molecule_instance 2" --region-label C
```
