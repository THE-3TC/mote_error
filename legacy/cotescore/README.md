# cotescore has been replaced by mote_error

`cotescore` is no longer maintained. Its successor is [`mote_error`](https://github.com/THE-3TC/mote_error):

```bash
pip install mote_error
```

```python
import mote_error
```

This final release of `cotescore` (0.4.0) contains no code. Installing it pulls in `mote_error`, and `import cotescore` raises an error that explains how to migrate.

## Migrating

The COTe score has been reframed as the **MOTe error** (Missing, Overlap, Trespass, with Excess as a support metric). With default weights:

- `MOTe = 1 − COTe`
- `Missing = 1 − Coverage`

so results you have already computed convert directly.

| cotescore | mote_error |
|---|---|
| `cote_score(...)` → `(COTe, Coverage, O, T, E)` | `mote(...)` → `(MOTe, Missing, O, T, E)` |
| `cote_class` | `mote_class` |
| `compute_cote_masks` | `compute_mote_masks` |
| `visualize_cote_states` | `visualize_mote_states` |
| `ClassCOTeResult` / `ClassCOTeCounts` | `ClassMOTeResult` / `ClassMOTeCounts` |

Everything else (`coverage`, `overlap`, `trespass`, `excess`, the OCR functions, adapters and dataset helpers) keeps its name.

## Still need the old code?

Pin the last real release:

```bash
pip install cotescore==0.3.0
```
