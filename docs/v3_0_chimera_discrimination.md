# v3.0 Chimera Discrimination

## The gap v1.0–v2.5 hid

Every release through v2.5 simulated a **single** banknote: `make_synthetic_fragments`
cut one note into non-overlapping Voronoi pieces. In that world every piece is
unique, the pieces tile the note exactly, and any non-overlapping high-coverage
set is the note. So the test suite was always green — but the failure mode that
makes this problem hard could not occur, because there was never a second note.

The engine encodes the same blind spot:

- `compute_compatibility` (and the fast/streaming variants) decide compatibility
  from **pixel overlap only**. Serial number, condition, and edge RGB are absent.
- `solve_covering_sets` accepts any set that is non-overlapping and covers the
  target area.

Together: **any non-overlapping set of pieces that tiles a note is a valid
solution, whether or not the pieces come from the same physical note.** With
2000 notes of one denomination, that is a chimera (缝合怪) factory.

## The testbed (the mirror)

`make_multi_note_fragments` builds the honest case: `notes` banknotes of one
denomination sharing **one region partition** (same layout, same cut positions),
each with its own appearance gain (wear / yellowing / ink density) and serial
number. All pieces are mixed into one pool; each piece carries its true
`note_id` in `meta` for diagnosis only.

`diagnose-chimeras` runs the **existing** solver on two matrices and counts
chimeras with `diagnostics.diagnose_solutions`:

```bash
moneyrepair simulate-multi-note --output runs/pool.npz --notes 5 --pieces-per-note 10
moneyrepair diagnose-chimeras --dataset runs/pool.npz --vis-dir runs/diag --output runs/diag.json
```

Measured result (5 notes, 10 pieces each, overlap-only matrix): **18 of 20
candidates are chimeras (90%)**, and only one true note is recovered pure. The
failure starts at N=2. This is the engine working exactly as written — the
simulation simply never exercised it before.

## The fix (discrimination in the matrix)

Add the dimension the matrix was missing. `fingerprint.fragment_appearance`
fits a per-channel gain `observed ≈ gain · template` over a fragment's masked
pixels. Because it is measured **relative to the standard template**, it cancels
the per-region content and recovers the note's tone transform regardless of
which region the fragment covers. `cluster_fragments_by_appearance` groups
fragments by that gain — one cluster per note.

`compute_compatibility_clustered` then marks a pair compatible only when the
fragments are **in the same group and** non-overlapping. `build-matrix
--discriminate appearance` (or `serial`, using serial labels) builds it.

Same pool, appearance-discriminated matrix: **0 chimeras, all 5 notes recovered
pure.** The work scales with per-note pair count, not `n²`.

## Honest limits (the residual hard tail)

This is not magic; it adds the missing axis and its power scales with how
distinguishable the notes are:

- Appearance discrimination separates notes by **condition/tone**. Two notes in
  near-identical condition fingerprint alike, so they can still cross-stitch.
  The residual chimera space shrinks from "all notes" to "notes of
  indistinguishable condition".
- That residual is where the v2.0 plan's "ambiguous residual fragments"
  assumption was wrong **at scale**: ambiguity is not a small tail when 2000
  notes share a denomination. The honest tools for the residual are serial OCR
  (`--discriminate serial`, already wired for labelled pieces) and the torn-edge
  contour similarity in `features.py`, which is **built but still not connected
  to the matrix**.

## Logged next steps (from external review)

- Connect `features.py` torn-edge contour similarity into the compatibility
  matrix for the same-condition residual.
- Input-side localisation: estimate each fragment's candidate pose(s) on the
  template instead of assuming `affine_to_note` is given; search candidates, not
  single placements.
- Performance for the 20k pool: avoid per-DFS-step `np.unpackbits` by keeping a
  pre-unpacked or bitset candidate set; parallelise the pairwise loop; optionally
  JIT the inner search.
