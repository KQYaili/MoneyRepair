# Pipeline Notes

MoneyRepair is built around one shared note coordinate frame.

## Data model

Each fragment has:

- `id`: stable internal id.
- `label`: physical or input label, usually from a file name or manifest field.
- `side`: `front` or `back`.
- `mask`: occupied pixels in note coordinates.
- `image`: sparse RGB pixels in note coordinates.
- `meta.affine_to_note`: optional 2x3 transform from local photo pixels to note
  coordinates.

## Real input path

The practical first path is a JSON manifest:

```json
{
  "note": {"width": 420, "height": 180},
  "fragments": [
    {
      "id": "frag-0001",
      "label": "0001",
      "side": "front",
      "image": "fragments/0001.png",
      "mask": "masks/0001.png",
      "affine_to_note": [[1, 0, 120], [0, 1, 36]]
    }
  ]
}
```

If `mask` is omitted, the loader infers foreground from the alpha channel or from
the corner background color. If `label` is omitted, the image filename stem is
used.

For one clear scan or photo containing many separated fragments, use
`segment-scan` first. It thresholds foreground against the corner background
color, extracts connected components, writes RGBA crops plus mask PNGs, and
generates an editable manifest. By default it preserves the original scan
coordinates as a simple translation affine; if the scan is only a staging photo,
edit the generated `affine_to_note` fields later after manual or automated
placement.

Use `label-manifest` after segmentation when labels need a second pass. It can
copy labels from a CSV file, derive them from filenames or ids, or call optional
Tesseract OCR. OCR is intentionally optional because it depends heavily on the
local executable and scan quality; the manifest format keeps the recognized
label editable either way.

## Compatibility evidence

The current matrix is pairwise same-note compatibility:

- mask overlap above tolerance means two fragments cannot be on the same note;
- optional reference RGB scoring can remove fragments whose affine placement or
  side assignment is obviously wrong;
- contour matching can provide an extra short list for difficult fragments.

The stored matrix uses `numpy.packbits`, so a 20,000 by 20,000 boolean matrix is
about 50 MB before `.npz` compression.

If pairwise comparison has already been recorded, import the two-column pair
list with `moneyrepair import-pairs`. The search command loads the packed matrix
directly, so it does not need to expand the full 20,000 by 20,000 matrix into a
dense boolean array.

`moneyrepair estimate-matrix --fragments 20000` reports about 381 MB for a dense
boolean matrix versus about 48 MB for the packed representation before `.npz`
compression.

Use `moneyrepair benchmark-synthetic` for local timing evidence. It runs the
deterministic synthetic pipeline and reports separate timings for simulation,
matrix construction, and DFS solving.

## Search

Search is depth-first over compatible fragments ordered by descending area. It
prunes when remaining candidates cannot reach the target coverage. The expected
interactive workflow is to generate a handful of high-coverage candidates, render
them, open the generated HTML report, and manually choose the plausible
reconstruction.

## Batch confirmation loop

The batch commands keep a small JSON state file:

- `batch-next` searches only fragments not already confirmed in earlier notes,
  writes candidates, and renders a report.
- `batch-confirm` stores the accepted candidate as a reconstructed note.
- `batch-reject` stores a sorted fragment-id key so the same bad candidate is
  skipped in later searches.

This mirrors the intended manual workflow: once an easy note is accepted, its
fragments disappear from the active set, so later searches have fewer choices.

## Honest Multi-Note Discrimination (v3.0)

When reconstructing fragments from a pool containing multiple banknotes of the same denomination (a multi-note pool), a pure pixel-overlap compatibility matrix is insufficient. Since identical-denomination notes share the same spatial design and templates, fragments from different notes can easily tile a template without overlapping, producing "chimera" (縫合怪) solutions that mix different physical notes (resulting in ~90% chimeras in standard 5-note pools).

To address this, MoneyRepair incorporates **Appearance Fingerprint Discrimination**:
- **Gain-Fitting**: Each fragment is matched against the reference template to estimate a per-channel brightness/color gain factor:
  $$observed \approx gain \times template$$
  This tone transform is invariant to the specific region of the banknote.
- **Clustering**: Fragments are grouped into clusters using a density-based algorithm (like DBSCAN) on their appearance gain vectors. Each cluster corresponds to a distinct physical note.
- **Discrimination Matrix**: `compute_compatibility_clustered` restricts compatibility. Two fragments are compatible only if they belong to the same appearance cluster (or share the same serial label in serial-based discrimination) and do not overlap.

## Production-Grade Auto-Locator & Candidate Pose Search (v4.0)

In a real-world scenario, approximate fragment placements are not pre-aligned. Instead, the pipeline automatically estimates multiple candidate poses for each fragment and searches over combinations of these poses.

### 1. Auto-Locator with Coarse-to-Fine Search (`locator.py`)
To align an input crop against the templates without given placement, the locator performs template matching over the front and back reference images:
- **Pyramid Downsampling**: The template and the crop are downsampled to Level 1 ($0.5\times$ resolution). The coarse global search scans the template using a step size of 8.
- **Numba JIT Acceleration**: The matching inner loop is decorated with `@numba.njit` utilizing zero-allocation flat array indexing. This eliminates Python interpreter and array allocation overhead, accelerating the search to **~67ms per fragment**.
- **Fine Refinement**: The Top-K candidate poses from the coarse search are scaled up to Level 0 ($1.0\times$ resolution), and a local $9\times9$ grid search is run to refine the position to the highest matching score.

### 2. Candidate Pose Solver Integration
- **Virtual Placed Fragments**: Each candidate pose (specifying X, Y, rotation, side, and match score) is represented as a virtual placed fragment with a unique ID format `f{piece_index}_pose{pose_index}`.
- **Mutual Exclusion Matrix**: When building the compatibility matrix, selecting pose $P_{i,j}$ for fragment $i$ must exclude all other poses of fragment $i$ from the candidate search pool. The matrix builder enforces this constraint by setting compatibility between different poses of the same fragment to `False`.
- **CLI Support**: The `--auto-locate` command-line argument triggers candidate pose search inside `run-pipeline`, allowing end-to-end reconstruction from raw unaligned fragment crops.

