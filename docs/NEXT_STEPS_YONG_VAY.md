# Next steps — Ng Yong Vay (Module 2 + leader)

Status as of 7 August 2026. Written after the Week 2 integration run.

A note on labels first. The plan calls you **Member B**; Xing Szen's
`ground_truth.py` header calls him **Member A**. Pick one convention and fix it
in the plan before the report is written, because the cover page, the Group
Contract Form and Appendix B all have to agree. Module ownership itself is not
in dispute: you have registration, localisation and integration.

## What is already done

The repository is scaffolded and the pipeline runs end to end. That was the
blocker for everyone: `ground_truth.py` was crashing not because of a bug in
it but because the `src/` and `data/` layout it expected did not exist yet.
Creating it was your Task 2.0. It now parses all 1,500 pairs and 10,013
defects.

- Task 2.0 repository setup — done
- Task 2.1 registration, ORB/SIFT + RANSAC with phase-correlation fallback — done
- Task 2.2 registration quality metric — done, and it now gates the warp
- Task 2.3 signed differencing — done ahead of schedule
- Task 2.4 blob extraction — done ahead of schedule
- Task 2.5 integration spine — running, with stubs for Modules 1 and 3
- 15 tests passing

Baseline over 88 pairs, localisation only: **F1 0.780 at IoU 0.5**,
0.153 s per board. Objective 1 wants 0.80.

## Do these this week

**1. Send the team the four findings.** They are in the README and each is
worth more than a week of independent tuning. In particular Xing Szen needs
finding 3 before he starts task 1.10, and Zhi Xuan needs finding 1 before he
writes stage one of the classifier — his stage one is already solved, it is
the polarity field on the blob.

**2. Raise the IoU threshold question at the checkpoint.** Objective 1 says
F1 ≥ 0.80 at IoU 0.5. The DeepPCB authors benchmark their own detector at
IoU 0.33. You are currently passing 0.780 at IoU 0.5, so keep the stricter
figure — it is a better objective — but report both in Chapter 4, otherwise
the comparison against published work is not like for like.

**3. Decide how the padding convention is reported.** This is the one that
could look bad if it surfaces in the viva rather than in the document. It is a
legitimate correction, but it must appear in Chapter 3 as a stated
methodological decision with the before-and-after numbers, not buried in a
constant. Write that paragraph yourself; it is Section 3.1, which you own.

**4. Freeze the contracts formally.** `src/contracts.py` exists and both other
modules import it. Get verbal agreement at the checkpoint and record it in the
meeting log, so that the file is a decision rather than your preference.

**5. Group Contract Form.** Section 6 of the plan already has the roles column
drafted. It needs three signatures and dates. Chase it now — it is Appendix A
and it is worth marks for existing.

## Then, in priority order

**Class-aware evaluation.** Current F1 ignores class labels. Extend
`score_pair` to require a class match, and you will get the real Objective 1
number. Expect it to drop sharply until Zhi Xuan's stage two exists. Better to
see that in Week 2 than in Week 6.

**Tune against the full test set, not a sample.** All the parameter choices so
far come from 30 to 88 pairs. `data/DeepPCB/PCBData/test.txt` lists the 500
held-out pairs. Tune on `trainval.txt` and report on `test.txt`, and say so in
Chapter 3 — tuning and reporting on the same images is the single easiest
methodological criticism to make of a project like this.

**Precision is the bottleneck, and it is Module 1's to fix.** Recall 0.741,
precision 0.823. The residual false positives are small blobs from
binarisation jitter, which is exactly what tasks 1.4 and 1.6 are for. Give
Xing Szen the harness so he can measure his filters against detection F1
rather than against how the images look.

**HRIPCB.** Nothing has been run against it. It is the justification for the
whole rectification requirement and for the robustness paragraph in Chapter 4.
It is also colour, higher resolution, and has no template images — so the
golden-template method needs a reference board synthesised or chosen. Raise
this early; it is a scope risk, not a task.

## Risks worth naming at the checkpoint

- **No template images in HRIPCB.** The system's core assumption does not hold
  for the secondary dataset. Decide in Week 3 whether HRIPCB is used for
  rectification demonstration only, and adjust Objective 2's wording if so.
- **Objective 1 is measured on localisation today.** Nobody should quote 0.780
  as the Objective 1 result until classes are scored.
- **Registration has no HRIPCB evidence yet.** It is verified on synthetically
  rotated DeepPCB boards, which is weaker evidence than real misorientation.

## Meeting agenda template

Thirty minutes, same slot weekly. Three questions each: what landed, what is
blocked, what is next. Record decisions with dates in `docs/MEETING_LOG.md` —
that log plus the commit history is the evidence if the Free-Rider Policy is
ever invoked.
