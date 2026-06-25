# v4.0 Production Banknote Reconstruction

v4.0 targets the industrialization of the reconstruction algorithm, moving from "approximate placement is given" to "placements are automatically estimated, searched via candidate poses, and pruned using robust descriptors".

## 1. Candidate Pose Model & Locator
Currently, the data model assumes a single pre-aligned placement `affine_to_note` per fragment. Real phone photos require estimating multiple potential poses.
* **Auto-locator (`locator.py`)**: Computes multi-scale, multi-rotation template matching (using normalized cross-correlation NCC, Lab space RMSE, and edge gradient similarity) over front/back reference templates to yield Top-K candidate poses for each fragment.
* **Candidate Pose Search**: The DFS engine searches combinations of candidate poses (e.g., `n{note_index}_f{piece_index}_pose{pose_index}`). Selecting pose $P_{i,j}$ excludes all other poses of fragment $i$ from the search pool.

## 2. Robust Color & Edge Matching
To handle phone photos with uneven lighting, shadows, and color casts:
* **Normalized Color Features**: Replaces raw RGB difference with luminance-normalized Lab color histograms and local gradient structure similarity.
* **Torn-edge Alignment**: Integrates `features.py` contour similarity matching into the compatibility matrix to penalize mismatching physical boundary interfaces.

## 3. Solver Bitset Pruning & JIT Acceleration
To scale search queries to 20,000+ candidates:
* **Pre-unpacked Matrix**: Stores compatibility matrix as a continuous, dense boolean array in memory before starting the DFS search to avoid per-recursive-step `np.unpackbits` overhead.
* **Bitwise Selection**: Performs candidate narrowing using fast numpy/C bitwise `&` operations:
  `next_candidates = current_candidates & compatibility_matrix[selected_pose_idx]`
* **Optional JIT Compiling**: Decorates the inner search loop using `@numba.njit` to eliminate Python runtime interpreter overhead.

## 4. Anchor Selection Logic
* **Anchor Scoring**: Automatically calculates an `anchor_score` for each fragment:
  `anchor_score = ocr_match_score + area_weight + uniqueness_bonus`
  Fragments containing high-confidence text (like serial numbers or denomination numbers), large areas, or low placement ambiguity (few candidate poses) score high.
* The batch confirmation engine starts DFS searches from the highest scoring anchor fragments to prune the search space early.
