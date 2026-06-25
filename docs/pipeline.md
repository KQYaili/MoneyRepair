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
