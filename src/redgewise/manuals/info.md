# redgewise info

Show program information, summarize a redgewise build output, and inspect
selector-relevant values.

## Synopsis

```bash
redgewise info
redgewise info -i /tmp/output --list
redgewise info -i /tmp/output --selector
redgewise info -i /tmp/output --list --selector --long
```

## Modes

- `--list`: print output summary and vertex table. This is the default when
  `--input` is supplied and neither `--list` nor `--selector` is used.
- `--selector`: print selector grammar and values observed in this output.
- `--long`: disable truncation.

## Selector inventory

With `--selector`, `redgewise info` prints fields and observed values such as:

```text
kind
molecule_type / moltype
molecule_instance / molinstance
residue_name / resname
atom_name / name
atom_type / type
residue_id / resid ranges
vertex_id / id ranges
```

## Compute selector examples

When `--input` is supplied, examples are generated from actual values in
`vertices.parquet`. Examples that cannot be instantiated for the current output
are omitted.

## Examples

```bash
redgewise info -i /tmp/output --selector
```

```bash
redgewise info -i /tmp/output --selector --long
```

```bash
redgewise info -i /tmp/output --list --long
```
