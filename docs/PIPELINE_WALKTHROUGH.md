# Pipeline walkthrough — one board, traced end to end

**Purpose.** A leader's reference to the whole system, not just one module. Every
number below was captured by instrumenting the real pipeline on a real board.
Nothing here is illustrative or from memory.

**Board traced:** `group00041/00041/00041000` — template and test pair
**Ground truth:** 10 annotated defects
**Result:** 9 detected, verdict FAIL, 55 ms total

Reproduce with:

```bash
python -m src.pipeline data/DeepPCB/PCBData/group00041/00041/00041000_temp.jpg \
                       data/DeepPCB/PCBData/group00041/00041/00041000_test.jpg
```

---

## The pipeline at a glance

`src/pipeline.py` is the orchestrator. It contains no image processing of its
own — its only job is to move data across module boundaries in the shapes fixed
by `src/contracts.py`. Read it first; it is 90 lines and it is the map.

| # | Stage | File | Owner | Status |
| --- | --- | --- | --- | --- |
| 1 | Load pair | `module1/preprocess.py` | Xing Szen | Real |
| 2 | Denoise | `module1/preprocess.py` | Xing Szen | **Stub** |
| 3 | Enhance | `module1/preprocess.py` | Xing Szen | **Stub — does nothing** |
| 4 | Binarise | `module1/preprocess.py` | Xing Szen | **Stub** |
| 5 | Calibrate | `module1/preprocess.py` | Xing Szen | **Stub — constant** |
| 6 | Contract check | `contracts.py` | Yong Vay | Real |
| 7 | Register | `module2/registration.py` | Yong Vay | Real |
| 8 | Signed difference | `module2/difference.py` | Yong Vay | Real |
| 9 | Morphological cleanup | `module1/morphology.py` | Xing Szen | Real |
| 10 | Blob extraction | `module2/blobs.py` | Yong Vay | Real |
| 11 | Classify, measure, verdict | `module3/classify.py` → `descriptors.py` | Zhi Xuan | Real |

Five of eleven stages are still stubs, and all five belong to Module 1.

---

## Step 1 — Load the pair

**File:** `src/module1/preprocess.py` → `load_pair()` · **Owner:** Chan Xing Szen

Reads both images from disk with `cv2.imread(..., IMREAD_GRAYSCALE)` and raises
`FileNotFoundError` if either fails, because `imread` returns `None` on failure
rather than raising, and a `None` propagating downstream produces a confusing
error several stages later.

**Output**

| | Template | Test |
| --- | --- | --- |
| Shape | 640 × 640 | 640 × 640 |
| Dtype | uint8 | uint8 |
| Distinct values | 16 | 76 |
| Mean intensity | 219.2 | 219.5 |
| Pure white | 85.2% | 84.6% |

> **Leader's note.** The template has only 16 distinct grey values. DeepPCB
> ships images that are already close to binary — the dataset authors
> pre-processed them. This is why Module 1's stubs work well enough for the
> pipeline to run today, and it is also why Objective 2's filter benchmark will
> show smaller differences than expected. Worth raising with Xing Szen before
> he designs the experiment.

---

## Step 2 — Noise removal

**File:** `src/module1/preprocess.py` → `denoise()` · **Status: stub (task 1.4)**

Currently a 3 × 3 median filter. Median is the placeholder because it removes
salt-and-pepper speckle without blurring trace edges. Task 1.4 must benchmark
Gaussian, median and bilateral against detection F1.

**Output:** same shape and dtype; **8,122 pixels changed** (2.0% of the image);
distinct values drop 16 → 15.

---

## Step 3 — Contrast enhancement

**File:** `src/module1/preprocess.py` → `enhance()` · **Status: stub (task 1.5)**

```python
def enhance(image, params=None):
    return image
```

**Output:** byte-for-byte identical to its input. This stage currently does
nothing at all.

> **Leader's note.** This is the clearest gap in the project. Histogram
> equalisation, CLAHE and linear stretching are named in the assignment's core
> functional requirements, and Table 2.2 of the report is a comparison of
> exactly these three. Nothing exists yet.

---

## Step 4 — Binarisation

**File:** `src/module1/preprocess.py` → `binarise()` · **Status: stub (task 1.6)**

Global Otsu thresholding. Otsu picks the threshold maximising between-class
variance, which suits a bimodal copper-against-substrate histogram.

**Output:** strictly `{0, 255}`, 86.0% white.
**Otsu chose a threshold of 6.**

> **Leader's note.** A threshold of 6 out of 255 means Otsu is separating
> "almost pure black" from "everything else" — confirmation that the input was
> already effectively binary. On genuinely greyscale imagery this threshold
> would land near 128. Do not let this pass without comment in Chapter 4; a
> marker who knows Otsu will ask.

---

## Step 5 — Spatial calibration

**File:** `src/module1/preprocess.py` → `calibrate()` · **Status: stub (task 1.7)**

```python
return 1.0 / DEEPPCB_PIXELS_PER_MM   # 48.0
```

**Output:** `mm_per_px = 0.0208333` (48 px/mm), a constant quoted from the
dataset README rather than derived from a measured board feature. Every area in
mm² downstream inherits this assumption. Already listed in report Section 5.2.

---

## Step 6 — Contract check

**File:** `src/contracts.py` → `PreprocessResult.validate()` · **Owner:** Ng Yong Vay

The boundary between Module 1 and Module 2. Checks four invariants: both images
the same shape, dtype `uint8`, values strictly in `{0, 255}`, and `mm_per_px > 0`.

**Output:** passed. Carries `template_bin`, `test_bin`, `mm_per_px`, `params`.

The point is that a Module 1 fault surfaces *here*, named, rather than as an
unexplained crash inside blob extraction.

---

## Step 7 — Registration

**File:** `src/module2/registration.py` → `register()` · **Owner:** Ng Yong Vay

Three candidate alignments compete, scored on the fraction of pixels that still
disagree after warping. Full explanation in the file's docstrings.

**Output — this is the interesting one:**

| | Value |
| --- | --- |
| Baseline disagreement (do nothing) | **0.715%** |
| Feature path: matches found | 268 |
| Feature path: RANSAC inliers | 212 (79.1%) |
| Feature path: reprojection residual | 0.946 px |
| Feature path: disagreement after warp | **0.738%** |
| **Method chosen** | **`identity`** |

The feature path *worked* — 268 matches, 79% inliers, sub-pixel residual. By
every conventional measure that is a good registration. And it still made the
images line up **worse** (0.738% vs 0.715%), so it was rejected.

> **Leader's note.** This single table is the best evidence in the project that
> a design decision was driven by measurement. It is Section 2.2 and Section
> 5.1 of the report. If one number gets quoted in the viva, make it this one.

---

## Step 8 — Signed difference

**File:** `src/module2/difference.py` → `signed_difference()` · **Owner:** Ng Yong Vay

Two bitwise operations that resolve half the classification problem:

```
removed = template_copper AND NOT test_copper    → open circuit, mouse bite, pin hole
added   = test_copper AND NOT template_copper    → short, spur, spurious copper
```

`copper_mask()` inverts first, because copper is *dark* in DeepPCB. Getting this
backwards inverts every class label while leaving the boxes looking perfect.

**Output:** two binary images, same shape.

| | Differing pixels | % of image |
| --- | --- | --- |
| `removed` | 1,695 | 0.41% |
| `added` | 1,235 | 0.30% |

---

## Step 9 — Morphological cleanup

**File:** `src/module1/morphology.py` → `clean_difference()` · **Owner:** Chan Xing Szen

Opening then closing with a 5 × 5 elliptical element. Called from the
orchestrator — the ordering belongs to the spine, the tuning belongs to Module 1.

**Output — the highest-leverage step in the whole pipeline:**

| | Before | After | Survived |
| --- | --- | --- | --- |
| `removed` pixels | 1,695 | 619 | 36.5% |
| `added` pixels | 1,235 | 734 | 59.4% |
| **Connected components** | **226** | **10** | **4.4%** |

226 candidate regions become 10. Everything discarded was binarisation jitter
along trace edges — long, one or two pixels wide, and unable to contain a 5 × 5
ellipse. Everything kept was compact.

> **Leader's note.** Without this step the system reports 226 defects on a board
> that has 10. This is a geometric argument, not a statistical one, and it is
> the single most quotable result Xing Szen owns.

---

## Step 10 — Blob extraction

**File:** `src/module2/blobs.py` → `extract_blobs()` · **Owner:** Ng Yong Vay

8-connectivity connected components on each polarity image separately, so every
region inherits the polarity of the image it came from. Regions under 40 px are
dropped as residue. Each blob carries its contour, because Module 3 needs the
outline for shape descriptors, not just the box.

**Output:** 10 components in → **9 blobs** out (one below 40 px).

| id | bbox (x, y, w, h) | area px | polarity | contour pts |
| --- | --- | --- | --- | --- |
| 0 | (437, 35, 9, 7) | 47 | removed | 15 |
| 1 | (502, 38, 14, 8) | 82 | removed | 19 |
| 2 | (550, 271, 29, 33) | 398 | removed | 59 |
| 3 | (260, 351, 9, 10) | 66 | removed | 15 |
| 4 | (161, 159, 12, 7) | 67 | added | 16 |
| 5 | (345, 263, 10, 9) | 51 | added | 15 |
| 6 | (468, 317, 16, 65) | 379 | added | 51 |
| 7 | (231, 324, 12, 12) | 101 | added | 12 |
| 8 | (99, 479, 19, 9) | 136 | added | 21 |

---

## Step 11 — Classify, measure, decide

**Files:** `src/module3/classify.py` (adapter) → `src/module3/descriptors.py` (rules)
**Owner:** Ng Zhi Xuan

Stage one is free: the polarity is already on the blob. Stage two computes
descriptors and applies threshold rules on aspect ratio and solidity. Area is
converted with `area_px × mm_per_px²`. The verdict fails the board above a
configurable defect count (default 0).

**Output:** 9 defects, verdict **FAIL**, area range 0.020–0.173 mm².

### How it did against the answer key

| id | predicted | ground truth | IoU | class correct |
| --- | --- | --- | --- | --- |
| 0 | mouse_bite | pin_hole | 0.73 | no |
| 1 | mouse_bite | pin_hole | 0.94 | no |
| 2 | open_circuit | open_circuit | 0.86 | **yes** |
| 3 | mouse_bite | pin_hole | 0.93 | no |
| 4 | spur | spurious_copper | 0.93 | no |
| 5 | spur | spur | 0.71 | **yes** |
| 6 | short | short | 0.78 | **yes** |
| 7 | spurious_copper | spur | 0.89 | no |
| 8 | spur | spurious_copper | 0.94 | no |

**Localisation: 9 of 10 found.** **Classification: 3 of 9 correct.**
**Polarity: 9 of 9 correct** — every error stays inside its polarity group.

That is the whole project's story on one board.

---

## A defect found while tracing this

`descriptors.extract_descriptors` computes solidity as:

```python
solidity = float(area_px) / hull_area          # area_px from connected components
                                               # hull_area from cv2.contourArea(hull)
```

These two quantities come from different geometries. `area_px` counts pixels;
`cv2.contourArea` measures a polygon through pixel *centres*, which
systematically underestimates the pixel count for small regions. The result is
solidity greater than 1, which is geometrically impossible — solidity is a ratio
of a shape to its own convex hull and cannot exceed 1.

**On this board, 7 of 9 blobs. Across 60 test pairs, 263 of 310 blobs — 85%.**

| id | area px | hull area | solidity | |
| --- | --- | --- | --- | --- |
| 0 | 47 | 38.5 | 1.221 | impossible |
| 1 | 82 | 73.5 | 1.116 | impossible |
| 2 | 398 | 450.0 | 0.884 | valid |
| 6 | 379 | 546.0 | 0.694 | valid |
| 7 | 101 | 85.5 | 1.181 | impossible |

**Why it matters.** The stage-two rule tests `solidity > 0.85`. Since solidity
exceeds 1 for 85% of blobs, that condition is almost always true, and the
classifier effectively decides on aspect ratio alone. This is a concrete,
named cause for the 29–63% within-group recall reported in Section 4.1.

**The fix is one line** — use the contour consistently for both:

```python
contour_area = cv2.contourArea(contour)
solidity = contour_area / hull_area if hull_area > 0 else 0.0
```

This is Zhi Xuan's file and his marks, so it should go to him as a finding
rather than be fixed for him. Raise it at the checkpoint and let him verify it.

---

## Where the time goes

Total ≈ 55 ms per board against Objective 3's 3,000 ms budget.

| Stage | ms | Share |
| --- | --- | --- |
| Load pair (disk I/O) | 30.40 | 55% |
| Registration | 15.17 | 28% |
| Signed difference | 3.88 | 7% |
| Blob extraction | 3.09 | 6% |
| Binarisation | 1.00 | 2% |
| Morphology | 0.72 | 1% |
| Classification | 0.20 | <1% |
| Denoise | 0.15 | <1% |
| Enhance | 0.001 | 0% |

> **Leader's note.** Reading the files from disk costs more than every algorithm
> combined. Objective 3 is met roughly fifty times over, so there is no case for
> optimising anything — and a Chapter 4 that spends space on performance tuning
> is spending it in the wrong place. If batch processing is added later, caching
> the template is the only optimisation worth making.

---

## What to take to the next checkpoint

1. **Module 1 is the critical path.** Three of its five stages do nothing real.
   Objective 2 has no evidence at all until 1.4, 1.5 and 1.6 exist.
2. **The solidity bug is the cheapest accuracy win available.** One line, and it
   plausibly moves the classification result materially.
3. **DeepPCB arrives near-binary.** This weakens the preprocessing study before
   it starts. Decide now whether Objective 2 is benchmarked on deliberately
   degraded images so the filters have something to remove.
4. **Registration and differencing are done and measured.** Module 2 is not the
   risk.
