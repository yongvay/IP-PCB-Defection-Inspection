# Weekly checkpoint log

Thirty minutes, same slot every week. Three questions each: what landed, what
is blocked, what is next. The leader records the outcome here on the day.

This log plus the commit history is the objective evidence of contribution if
the TAR UMT Free-Rider Policy is ever invoked, so it is filled in as the term
runs rather than reconstructed at the end. An entry written six weeks late is
worth very little.

**Standing rule.** Nobody changes an interface in `src/contracts.py`
unilaterally. Amendments are raised here, agreed, and then applied by the
leader.

---

## Template — copy this block for each meeting

### Week N — DD Month 2026

Present: ·
Apologies: ·

| Member | What landed | What is blocked | What is next |
| --- | --- | --- | --- |
| Chan Xing Szen | | | |
| Ng Yong Vay | | | |
| Ng Zhi Xuan | | | |

**Decisions**

| # | Decision | Raised by | Agreed |
| --- | --- | --- | --- |
| | | | |

**Actions**

| Action | Owner | Due |
| --- | --- | --- |
| | | |

---

## Week 1 — DD August 2026

*Not yet recorded. The kick-off items below are outstanding and should be*
*closed at the next checkpoint.*

**Outstanding kick-off items**

| Item | Owner | Status |
| --- | --- | --- |
| Lock the three SMART objectives (Section 1.2) | Ng Yong Vay | Drafted, not formally agreed |
| Freeze `src/contracts.py` as a group decision | Ng Yong Vay | File exists and is imported by both other modules; verbal agreement not recorded |
| Group Contract Form signed and dated by all three | All | Not started — this is Appendix A and carries marks for existing |
| AI Usage Disclosure Form started by each member | Each individually | Not started |
| Agree the reporting convention for the DeepPCB annotation padding | Ng Yong Vay | Raised in `NEXT_STEPS_YONG_VAY.md`, not yet discussed |

---

## Open questions carried between meetings

These are the items that need a recorded group decision. Each one changes
either the report or the schedule, so none should be settled privately.

1. **IoU threshold reported in Chapter 4.** Objective 1 sets IoU 0.5; the
   DeepPCB authors benchmark at IoU 0.33. Both should be reported so the
   comparison against published work is like for like.
2. **Annotation padding.** DeepPCB boxes are drawn with roughly a 10-pixel
   margin around each defect. The same convention is applied to predictions
   when scoring. This must appear in Chapter 3 as a stated methodological
   decision with before-and-after figures, not buried in a constant.
3. **HRIPCB has no template images.** The golden-template method's core
   assumption does not hold for the secondary dataset. Decide whether HRIPCB
   is used for rectification demonstration only, and reword Objective 2 if so.
4. **Member labelling.** The plan calls Ng Yong Vay "Member B" while
   `ground_truth.py` calls Chan Xing Szen "Member A". The cover page, Group
   Contract Form and disclosure forms must all agree. Pick one convention.
5. **Confidence score on the Defect contract.** The stage-two rules are hard
   thresholds and produce no graded score, so `confidence` is currently a
   fixed placeholder. Either derive a real value or remove the field.
6. **Physical dimensions on the Defect contract.** The dashboard and the PDF
   report show width and height in millimetres derived from the axis-aligned
   bounding box. Surfacing the rotated minimum-area box would need a contract
   amendment.
