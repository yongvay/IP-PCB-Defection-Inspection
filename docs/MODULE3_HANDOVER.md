# Module 3 handover — Ng Zhi Xuan

Defect Classification, Measurement & Analysis Dashboard. Owns SMART Objective 3.

This note records what was built, why each decision was taken, what was
measured, and what remains. Read it before the viva: understanding of code is
assessed individually and live, and every design choice below is one a marker
can reasonably ask about.

---

## 1. Files

| File | Task |
| --- | --- |
| `src/module3/descriptors.py` | 3.1 region descriptors, plus the descriptor-only baseline classifier |
| `src/module3/connectivity.py` | 3.2 trace-context classifier — the module's main contribution |
| `src/module3/classify.py` | 3.2 to 3.4: classifier selection, physical measurement, board verdict |
| `src/module3/evaluate.py` | 3.7 evaluation harness |
| `src/module3/pdf_report.py` | PDF export (extra-effort reporting requirement) |
| `dashboard.py` | 3.5 and 3.6 Streamlit dashboard |
| `tests/test_module3.py` | 28 regression tests, no dataset required |
| `experiments/benchmark_module3.py` | Every Chapter 4 number for this module |
| `tools/verify_module3.py` | Task-by-task self-check |
| `tools/diagnose_polarity.py` | Polarity diagnostic (see section 5) |

Two files belonging to Ng Yong Vay were edited. Both changes are additive and
are raised formally in section 7.

| File | Change |
| --- | --- |
| `src/contracts.py` | Six new fields on `Defect`, four on `InspectionReport`, all defaulted |
| `src/pipeline.py` | Builds the two copper masks for Module 3; makes copper polarity a parameter |

---

## 2. The central idea

The original classifier decided a defect's class from the **shape** of the
changed pixels: elongated regions became open circuits or shorts, compact
convex regions became pin holes or spurious copper, everything else fell
through to mouse bites and spurs.

That approach cannot work, and the reason is structural rather than a matter of
tuning. Three pairs of classes are shape-identical by definition:

- an open circuit and a mouse bite are both a piece missing from a trace;
- a short and a spur are both copper attached to a trace;
- a pin hole and a small spurious island are both compact and roughly round.

What separates them is not the shape of the changed region but **how that region
sits against the surrounding copper**. The DeepPCB taxonomy is itself
relational:

| Class | Relationship to the surrounding copper |
| --- | --- |
| Missing hole | Removed copper fully enclosed by remaining copper |
| Open circuit | Removed copper separating a trace into two ends |
| Mouse bite | Removed copper open to the substrate on one side |
| Spurious copper | Added copper touching no existing trace |
| Short | Added copper joining two otherwise separate traces |
| Spur | Added copper protruding from a single trace |

`connectivity.py` reads that relationship with classical morphology and no
machine learning. A ring is dilated around the candidate region, intersected
with the copper mask, and the separate contact patches are counted. Zero, one,
two-or-more and fully-enclosed are four distinct outcomes that map one-to-one
onto the taxonomy above.

**Which copper mask to read is the detail that is easy to get backwards.** For
*removed* copper the question is what the trace looks like after the defect, so
the **test** board's copper is the reference. For *added* copper the question is
what the stray metal touches that was legitimately there, so the **template**
copper is the reference — measuring against the test board would count the added
copper as part of the trace and make every spur look isolated.

The descriptor baseline is kept rather than deleted. A measured comparison
between two approaches is worth far more in Chapter 4 than a single unexplained
accuracy figure, and both are selectable at run time via
`params["classifier"]`. The dashboard offers the switch in the sidebar, so the
comparison can be demonstrated live.

---

## 3. Results

All figures at IoU 0.50. DeepPCB is the held-out test split; parameters were
chosen on trainval.

| Dataset | Localisation F1 | Descriptor class acc. | Connectivity class acc. |
| --- | --- | --- | --- |
| DeepPCB test | 0.8182 | 0.4583 | **0.9444** |
| HRIPCB upright | 0.6848 | 0.2159 | **0.5455** |
| HRIPCB rotated | 0.3508 | 0.1954 | **0.5517** |

On DeepPCB, classification F1 rises from 0.3750 to 0.7727 and macro F1 from
0.3285 to 0.7761. Runtime is 0.028 s per board against the three-second budget
in Objective 3; 100% of boards are within budget on all three datasets.

**Localisation F1 is identical across both classifiers, to four decimal
places, in every experiment run.** That is the validity check built into the
harness: it proves nothing but the classifier changed, so the entire
improvement is attributable to Module 3 and not to preprocessing, registration
or differencing. The check has now held across the classifier comparison, both
ring sweeps and both contact sweeps.

### Rotation invariance

Between upright and rotated HRIPCB, localisation F1 halves (0.6848 to 0.3508)
while class accuracy does not move (0.5455 to 0.5517). The collapse is almost
entirely precision (0.693 to 0.238); recall barely shifts (0.677 to 0.669), so
rectification generates false positives rather than losing genuine defects.

Classification survives because contact counting is a **topological** property,
not a geometric one. Rotating a board does not change how many separate copper
regions touch a defect. The descriptor baseline, reading aspect ratio and
circularity, has no such protection.

### The remaining failure mode

On DeepPCB, 20 of 360 matched detections are misnamed, and 12 of those are
mouse bites called open circuits. The boundary is physically real rather than a
coding error: a mouse bite deep enough to nearly sever a trace leaves copper on
both sides of the ring, so the contact count reads 2. The taxonomy is
continuous at that boundary.

Note also that per-class F1 conflates detection with classification. Mouse bite
recall of 0.28 and spur recall of 0.59 are dominated by defects the pipeline
never found — 63 of 107 mouse bites and 29 of 73 spurs appear in the
`background` column. Those are Module 1 and 2 limitations being charged to
Module 3's per-class table, which is exactly why the conditional class accuracy
figure exists.

---

## 4. Parameter justification

Both free parameters were swept on **trainval** and reported on **test**.

**Ring width.** The trainval curve has a clean interior maximum at 4 px
(class accuracy 0.9859), rising from 2 px and falling steadily to 20 px.
`RING_WIDTH_PX = 4` is therefore justified rather than assumed.

An earlier sweep on the *test* split rose monotonically over the same interval
and would have pushed the value to 8. Eighty boards give roughly 360 matched
detections, so the difference between 0.944 and 0.972 is about ten detections.
The honest conclusion is that the optimum is **broad and flat between 3 and
8 px** and differences inside that band are not resolvable at this sample size.
This is also a concrete demonstration of why tuning on the test split is
avoided: it would have selected a value trainval says is worse.

**Contact area.** The trainval curve is a plateau, not a peak: 2, 6 and 12 px
all score 0.9859 to 0.9819, then performance falls sharply at 25.
`MIN_CONTACT_PX = 6` sits mid-plateau, which is the robust choice.

**Neither parameter transfers between datasets, and they move in opposite
directions.** DeepPCB prefers a wide ring (4 px) with a low area threshold
(6 px); HRIPCB prefers a narrow ring (2 px) with a high one (50 px). The cause
is alignment quality and copper density rather than resolution: DeepPCB is
pre-aligned with clean binarisation, so a wide ring reliably reaches both trace
ends, whereas HRIPCB carries residual misregistration and denser layout, where
a wide ring straddles neighbouring unrelated traces and a low threshold admits
binarisation fragments.

The HRIPCB sweeps are **sensitivity analysis, not tuning**. The reported HRIPCB
figure of 0.5455 uses the DeepPCB-tuned parameters unchanged. Re-tuning would
recover roughly nine points (0.6364), and reporting the untuned figure is the
honest generalisation claim.

---

## 5. A bug found in shared code

On HRIPCB the system initially localised acceptably (F1 0.6848) but named
almost nothing correctly (class accuracy 0.0227). Those two figures cannot both
be explained by a weak classifier: stage one splits the six classes into two
disjoint families, so a classifier guessing at random *within the correct
family* would still score around 0.33.

`src/module2/difference.py` set `COPPER_IS_DARK = True` as a module constant.
That is correct for DeepPCB, which is binarised so copper falls dark. HRIPCB is
colour photography of green boards thresholded with adaptive mean, where copper
falls light. Stage one was inverted across the entire dataset.

`tools/diagnose_polarity.py` measured it directly, comparing the polarity
assigned to each matched detection against the polarity its ground-truth label
implies: **100% agreement on DeepPCB, 5% on HRIPCB**. After making
`copper_is_dark` a per-dataset parameter, HRIPCB agreement rose to 95% and
class accuracy from 0.0227 to 0.5455.

**Why this matters beyond the fix.** The fault was invisible to every metric
Modules 1 and 2 report. Inverting the polarity renames the two families without
changing which pixels differ, so localisation F1 was unaffected. It became
visible only once classification was scored separately from localisation —
which is the harness built for task 3.7. This is the strongest available
argument for the reporting decision in section 6.

It is also a limitation of the design: a global constant encoding "copper is
dark" cannot hold across two datasets with different imaging, and nothing in
the architecture forced anyone to notice.

---

## 6. What the evaluation harness reports, and why

`evaluate.py` separates three questions that a single F1 would conflate:

- **Localisation F1** — did the pipeline find the defect, whatever it called
  it. This measures Modules 1 and 2.
- **Classification F1** — did it also name it correctly. The gap between this
  and localisation is Module 3's error and nobody else's.
- **Conditional class accuracy** — of the defects that were found, what
  fraction were named correctly. This is the classifier's own score, and it is
  the figure to quote when comparing the two classifiers, because it does not
  move when Module 1 improves.

Chapter 4 reports all three. A single combined figure lets a weak classifier
hide behind strong detection, and vice versa.

Also reported: a confusion matrix with an explicit `background` row and column
so false positives and missed defects stay accounted for and the rows and
columns sum to the true totals; per-class precision, recall, F1 and **support**;
macro-averaged F1 alongside the micro average, because the class distribution
is uneven and a micro average can look healthy while a rare class is never
detected; runtime against the three-second budget; and CSV export of four
tables per run.

`score_pair` keeps its original signature, because `benchmark_pipeline.py` and
`test_integration.py` both call it. It is now a thin wrapper over `match_boxes`,
so the two code paths cannot disagree about what counts as a match.

---

## 7. Raise at the next checkpoint

Neither of the following is recorded in `docs/MEETING_LOG.md`. The working
agreement is that nobody changes `src/contracts.py` unilaterally, and the
polarity fix touches shared behaviour.

**Amendment to `Defect`** — adds `polarity`, `width_mm`, `height_mm`,
`severity`, `decided_by`. Every field carries a default, so existing code
constructing a `Defect` with the original five arguments is unaffected. Closes
open questions 5 and 6.

**Amendment to `InspectionReport`** — adds `mm_per_px`, `verdict_reason`,
`verdict_detail`, `classifier`. Same reasoning.

**`src/pipeline.py`** — builds the two copper masks with
`difference.copper_mask` and passes them to `build_report`. Built in the
pipeline rather than inside Module 3, because normalising which pixel value
means copper is Module 2's responsibility and is already solved there.

**`copper_is_dark` as a parameter** — defaults to the existing constant, so
DeepPCB behaviour is byte-identical. Only the HRIPCB benchmark overrides it.
This is a correctness fix to shared code and needs minuting as such, not
leaving as a commit message.

**Member labelling** (open question 4) is still unresolved and blocks the cover
page, the Group Contract Form and Appendix B.

---

## 8. What was fixed along the way

**The dashboard did not start.** It imported `DEEPPCB_PIXELS_PER_MM` from
`src.module1.preprocess`, where that constant does not exist; it lives in
`src.module1.calibration`. The dashboard now reads `report.mm_per_px`, so it
uses the calibration factor the pipeline actually applied.

**Physical dimensions came from the wrong box.** Width and height were taken
from the axis-aligned bounding box, so a 2 mm open circuit at 45 degrees was
reported as roughly 1.4 mm by 1.4 mm. They now come from the rotated
minimum-area rectangle.

**`confidence` was hard-coded to 0.5.** It is now the normalised margin by
which the deciding descriptor cleared its threshold, on a 0.5 to 1.0 scale. It
is a decision margin, **not** a probability, and Chapter 4 must not describe it
as one.

**The verdict was a bare defect count**, treating one short as equivalent to
one cosmetic nick. Task 3.4 now applies an ordinal severity weight (3 for an
open circuit or short, 2 for spurious copper, 1 for the rest), fails a board
outright on any critical defect regardless of tolerance, and returns the
deciding condition alongside the verdict.

**`process_blobs` was dead code**, and has been removed. Two classification
paths would leave a marker unable to tell which produced the results.

**A latent bug in the sweep.** `measure_context` originally took
`ring_px: int = RING_WIDTH_PX`. A default argument binds once when the function
is defined, so the sweep would have silently measured the same width nine times
and produced a flat curve that meant nothing. Both free parameters are now
resolved at call time.

**Missing descriptors.** `perimeter_px` and `eccentricity` were named in the
workload plan but not computed. Eccentricity is derived from the second-order
central moments rather than `cv2.fitEllipse`, which needs at least five contour
points and raises on the small regions a pin hole produces.

---

## 9. Known limitation, documented rather than fixed

`classify_by_connectivity` is asymmetric between polarities. The removed branch
returns `None` when the copper context cannot be read, and the caller falls
back to the descriptor rules. The added branch does not: `contacts == 0` is the
legitimate signature of spurious copper, so at extreme thresholds every added
defect is labelled spurious copper instead of falling back.

This is visible in the contact sweep, which saturates at 0.3139 on DeepPCB —
*below* the descriptor baseline of 0.4366, because the added family has
collapsed to a single class. It only occurs at settings far outside any
operating range, so it is documented rather than changed, but it is a genuine
design asymmetry.

---

## 10. Questions to expect in the viva

- Why is the area converted with `mm_per_px` **squared**? Area scales with the
  square of a linear factor; applying it once under-reports by a factor of 48
  on a 48 px/mm board.
- Why the *rotated* rectangle rather than the bounding box? A diagonal defect's
  bounding box measures the image axes, not the defect.
- Why read the **test** copper for removed defects and the **template** copper
  for added ones? Section 2.
- Why does the confusion matrix have a `background` row and column? So false
  positives and missed defects are accounted for and the totals reconcile.
- Why greedy matching rather than optimal assignment? The two agree on boards
  carrying 3 to 12 well-separated defects, and greedy is what the DeepPCB
  benchmark script uses, keeping the published comparison like for like.
- Why is `confidence` floored at 0.5 rather than 0? Because the class was still
  assigned. Reporting near-zero confidence for a decision the system acted on
  would misdescribe what happened.
- Why keep a classifier shown to be worse? The comparison is the result.
- How was the polarity bug found, given localisation looked fine? Section 5.
- Why report the untuned HRIPCB figure rather than the better one? Section 4.
