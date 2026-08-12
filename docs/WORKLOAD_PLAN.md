# **BMDS2133 Image Processing — Workload Distribution Plan**

**Project:** Automated PCB Defect Inspection System **Mode:** Mode B — Innovative Solution Development **Stack:** Python 3.11 \+ OpenCV \+ NumPy \+ Streamlit **Team:** Ng Yong Vay (Leader), Chan Xing Szen, Ng Zhi Xuan **Tutor:** Assoc. Prof Ts. Dr Tan Chi Wee **Prepared:** 1 August 2026

## **0\. Decisions made before splitting the work**

These are locked first, because every task below depends on them. Confirm them at the kick-off meeting.

### **0.1 Recommended defect scope**

Detect the **six standard PCB bare-board defect classes**:

| \# | Defect class | Nature of change vs reference board |
| ----- | ----- | ----- |
| 1 | Open circuit | Copper **removed** — trace broken |
| 2 | Mouse bite | Copper **removed** — bite out of trace edge |
| 3 | Missing hole / pin-hole | Copper **removed** — hole absent or void present |
| 4 | Short | Copper **added** — two traces bridged |
| 5 | Spur | Copper **added** — unwanted protrusion from trace |
| 6 | Spurious copper | Copper **added** — isolated stray copper island |

**Why these six and not component misplacement.** They are the taxonomy used by both recommended datasets, which means every defect has a **ground-truth bounding box and label**. That is what makes your SMART(**S**pecific, **M**easurable, **A**chievable, **R**elevant, **T**ime-bound) objectives measurable — you can compute precision, recall, F1 and IoU in Chapter 4\. Component-misplacement defects have no labelled ground truth in any public dataset, so the group would have to hand-label, and Chapter 4 would collapse into screenshots with no metrics. The rubric penalises exactly that ("Results are presented but lack meaningful experimental data").

**Why this is a strong classical image-processing problem.** The six classes split cleanly into *copper-added* and *copper-removed*, which is recoverable from the sign of the template–test difference. That gives a two-stage rule-based classifier built entirely from morphology and region descriptors — no machine learning needed, and no risk of being marked down for "lacking fundamental Image Processing algorithms".

### **0.2 Recommended dataset**

**Primary — DeepPCB** (Tang, He, Liu & Li, 2019). 1,500 image pairs; each pair is a 640 × 640 defect-free **template** image plus a **defective test** image, pre-aligned by template matching, \~48 pixels per millimetre, 3–12 annotated defects per image, six classes. The template/test pairing is the single most important property: the project scope says "validating board integrity **against a reference model**", and this dataset hands you the reference model directly.

**Secondary — HRIPCB / PKU-Market-PCB** (Huang & Wei, 2020). 1,386 images (693 original), colour, higher resolution, same six classes with bounding boxes. Crucially, **half the images show the board randomly oriented on the workbench**. Use this as the robustness test set — it is the justification for the Image Calibration and rectification requirement, and it produces an excellent "our system degrades by X% under rotation" paragraph in Chapter 4\.

**Compliance note:** the spec forbids uploading external web-based datasets. Cite them in Chapter 3.2 with URL and DOI instead. Include the DOI for HRIPCB (10.1049/joe.2019.1183).

### **0.3 Draft SMART objectives (Section 1.2 — write these in Week 1\)**

Everything in the report is graded against these, so agree them before any code is written.

1. **To develop** a golden-template-based PCB inspection pipeline that localises and classifies the six standard bare-board defect classes on the DeepPCB test set, achieving a mean F1-score of **at least 0.80** at an IoU threshold of 0.5, by the end of Week 6\.  
2. **To determine**, through controlled experiments, the best-performing combination of noise-removal filter and binarisation method for PCB imagery by benchmarking **at least three filters and two thresholding techniques** against detection F1-score, by the end of Week 4\.  
3. **To deliver** an interactive inspection dashboard that processes a template–test image pair and returns an annotated defect report with per-defect physical measurements in millimetres in **under 3 seconds per board**, by the end of Week 7\.

Each objective is owned by a different member (see Section 1), which is deliberate — it makes each person's Chapter 4 contribution self-evident.

## **1\. Module ownership**

Three self-contained modules, one per member, connected by fixed data contracts (Section 2). Each member can code, test and write about their module without waiting for the others.

### **Member A — CHAN XING SZEN**

**Module 1: Data Acquisition, Preprocessing & Calibration** *Owns SMART Objective 2\.*

| Task | Detail |
| ----- | ----- |
| 1.1 Dataset acquisition | Download DeepPCB ([https://github.com/tangsanli5201/DeepPCB/tree/master](https://github.com/tangsanli5201/DeepPCB/tree/master)) and HRIPCB, verify integrity, document licences and citations |
| 1.2 Ground-truth parser | Parse annotation files into a common Python structure the whole team uses |
| 1.3 Image ingestion | Loader for single pairs and folder batches; greyscale conversion; format handling |
| 1.4 Noise removal study | Implement and benchmark **Gaussian, median and bilateral** filtering; tune kernel sizes |
| 1.5 Contrast enhancement | Implement histogram equalisation, **CLAHE** and linear contrast stretching; compare |
| 1.6 Binarisation | Implement global **Otsu** vs **adaptive (mean and Gaussian)** thresholding; compare |
| 1.7 Spatial calibration | Derive the pixel-to-millimetre scale factor; expose it so defect areas report in mm² |
| 1.8 Geometric rectification | Perspective/rotation correction for the misoriented HRIPCB boards |
| 1.9 Unit tests | Test suite proving the pipeline is deterministic and handles edge cases |
| 1.10 Morphological cleanup | Opening/closing/tophat on the difference image to suppress registration jitter and scan noise without erasing genuine small defects. Clean contract: binary difference in → cleaned binary difference out |
| 1.11 Result figures | Generate every chart used in Chapter 4 from Zhi Xuan's metrics CSV |

**Report sections owned:** 1.1 Problem Background · 1.3 Motivation (SDG 9 / Malaysia MADANI) · 2.1 Preprocessing & Enhancement Algorithms · **Table 2.1** (Segmentation comparison) · **Table 2.2** (Contrast Enhancement comparison) · 3.2 Description of Dataset · 5.2 Limitations & Future Works · final UK-English proofread of the whole document.

---

### **Member B — NG YONG VAY**

**Module 2: Registration & Defect Localisation \+ System Integration** *Owns SMART Objective 1\.*

| Task | Detail |
| ----- | ----- |
| 2.0 Repository setup | GitHub repo, branch policy, folder structure, `requirements.txt`, README |
| 2.1 Image registration | Align test image to template using **ORB/SIFT feature matching \+ RANSAC homography**, with phase correlation as the fallback method |
| 2.2 Registration quality metric | Report alignment residual so a badly registered pair is flagged rather than silently producing false defects |
| 2.3 Golden-template differencing | Signed difference (`template AND NOT test` \= copper-removed; `test AND NOT template` \= copper-added) |
| 2.4 Blob extraction | Connected-component analysis, contour extraction, bounding boxes, centroids |
| 2.5 Integration spine | The `pipeline.py` orchestrator that calls Module 1 → Module 2 → Module 3 |
| 2.6 Code review | Review and merge all pull requests; enforce commenting and modularity (30% of prototype marks) |

**Report sections owned:** 1.2 Objectives · 2.2 Image Registration & Template Matching · **3.1 System Flowchart** · 5.1 Achievements · Appendix A Group Contract Form.

**Leadership duties (not module work, but real work — roughly 10 hours):** chair the weekly checkpoint and keep the log, own and assemble the master document, arbitrate interface changes, perform the final submission (the spec states only the group leader submits; the other two mark as done in Google Classroom).

**Why this looks smaller than the others on paper.** It is not. Registration (2.1) is the hardest single algorithm in the project, the integration spine (2.5) is the task that absorbs everyone else's slippage, and code review plus leadership overhead is roughly 18 hours that never appears in a module list. The content load is deliberately trimmed to make room for it.

---

### **Member C — NG ZHI XUAN**

**Module 3: Defect Classification, Measurement & Analysis Dashboard** *Owns SMART Objective 3\.*

| Task | Detail |
| ----- | ----- |
| 3.1 Region descriptors | Extract area, perimeter, aspect ratio, solidity, extent, eccentricity and **Hu moments** for each defect blob |
| 3.2 Two-stage classifier | Stage 1: copper-added vs copper-removed from the difference polarity. Stage 2: rule-based discrimination into the six classes using the descriptors above plus trace-connectivity context |
| 3.3 Physical measurement | Convert areas and dimensions to mm² / mm using Module 1's calibration factor |
| 3.4 Board verdict logic | Pass/fail decision against a configurable defect-count and severity tolerance |
| 3.5 Streamlit dashboard | Template \+ test upload, parameter sliders, side-by-side view, annotated overlay with colour-coded boxes and labels |
| 3.6 Summary panels | Defect count per class, area distribution, verdict banner, sortable defect table |
| 3.7 Evaluation harness | IoU matching against ground truth; per-class **precision, recall, F1**; confusion matrix; runtime benchmarking. Exports a metrics CSV for Xing Szen to chart |

**Report sections owned:** 2.3 Morphological & Region-Based Classification (the template's "2.x — add more if needed" slot) · 3.3 Applications of the Algorithms (compiles pseudocode from all three) · **4.1 Experimental Results** · **4.2 Discussion & Critical Interpretation** · full APA reference list, alphabetised.

**Template numbering note.** The provided template lists 2.1, 2.2, "2.x — add more if needed", then "2.3 Analysis of the existing algorithm (Comparison)". Renumber the comparison section to **2.4** once the third algorithm section is added, so the numbering runs 2.1 → 2.2 → 2.3 → 2.4. Both comparison tables (2.1 Segmentation, 2.2 Contrast Enhancement) live inside that final section.

## 

## **2\. Interface contracts — agree these in Week 1, before anyone codes**

This is the part most student groups skip and then lose a week to merge conflicts. Fix the data shapes first; then all three of you can build against stubs in parallel.

**Module 1 → Module 2**

PreprocessResult \= {

    "template\_bin": np.ndarray,   \# uint8, binary 0/255

    "test\_bin":     np.ndarray,   \# uint8, binary 0/255, same shape as template

    "mm\_per\_px":    float,        \# spatial calibration factor

    "params":       dict          \# filter/threshold settings actually used

}

**Module 2 → Module 3**

Blob \= {

    "id":       int,

    "bbox":     (x, y, w, h),     \# pixels, top-left origin

    "contour":  np.ndarray,

    "centroid": (cx, cy),

    "area\_px":  int,

    "polarity": "added" | "removed"   \# sign of the template-test difference

}

LocalisationResult \= {"blobs": \[Blob\], "align\_residual": float}

**Module 3 output**

Defect \= {

    "id": int, "bbox": (x,y,w,h), "class": str,

    "area\_mm2": float, "confidence": float

}

InspectionReport \= {"defects": \[Defect\], "verdict": "PASS"|"FAIL", "runtime\_s": float}

**Shared utility — `morphology.py` (owned by Chan Xing Szen)**

Morphological cleanup sits between differencing and blob extraction, so it is exposed as a single pure function that the orchestrator calls. Xing Szen owns the implementation and tuning; Yong Vay only calls it.

def clean\_difference(diff\_bin: np.ndarray, params: dict) \-\> np.ndarray:

    """Opening/closing/tophat. Same shape in, same shape out. No side effects."""

**Rule:** nobody changes a contract unilaterally. Raise it at the weekly meeting; the leader updates the shared spec.

## 

## **3\. Documentation ownership vs the marking rubric**

Documentation is 100 marks. Every criterion has exactly one accountable owner.

| Rubric criterion | Marks | Primary owner | Contributors |
| ----- | ----- | ----- | ----- |
| Introduction | 10 | **Xing Szen** (1.1 background, 1.3 motivation) | Yong Vay (1.2 objectives) |
| Literature Review | 20 | **Xing Szen** (2.1 \+ both comparison tables) | Yong Vay (2.2), Zhi Xuan (2.3) |
| Methodology | 20 | **Yong Vay** (3.1 flowchart) | Xing Szen (3.2 dataset), Zhi Xuan (3.3 pseudocode) |
| Result & Discussion | 20 | **Zhi Xuan** (4.1, 4.2) | Xing Szen (all figures), all supply their module's experiment |
| Conclusion | 10 | **Yong Vay** (5.1 achievements) | Xing Szen (5.2 limitations & future works) |
| Spelling, Grammar & UK English | 10 | **Xing Szen** (final proofreader) | all |
| References (APA, alphabetical) | 10 | **Zhi Xuan** | all supply their own citations |

Note that 1.2 Objectives stays with the leader even though the rest of Chapter 1 moves. The SMART objectives are what Chapters 4 and 5 are graded against, so the person assembling the master document must own them.

**Three writing rules that cost marks if broken** — the rubric is explicit about all three:

1. **UK English only.** analyse, colour, programme, centre, optimise, recognise. Set your word processor to English (UK) on day one.  
2. **Zero first-person pronouns.** No "I", "we", "our", "my". Write "The system applies…", "This study compares…". The top band requires *zero* slips; 1–2 slips already drops you a band.  
3. **No numbered lists in the References section**, entries alphabetised, avoid ResearchGate/arXiv/forums as sources — use the university library databases and journal publishers instead.

### **Prototype rubric ownership**

| Criterion | Weight | Accountable |
| ----- | ----- | ----- |
| User Interface / Output | 20% | Zhi Xuan |
| Programming (clean, modular, commented) | 30% | Yong Vay (enforced via code review; all three write their own) |
| Degree of Completion | 20% | Yong Vay |
| System Implementation & Integration | 20% | Yong Vay |
| Understanding of Code / On-the-spot Coding | 10% | **Each member individually** — you are quizzed on your own module |

**Warning on the last row:** it is assessed individually and live. Do not let anyone hand their module to someone else to finish. A member who cannot explain their own code loses those marks personally.

## 

## **4\. Schedule**

Your lecturer has not announced the deadline yet, so this is expressed in **relative weeks**. Anchor Week 1 to the Monday after the deadline is announced, and confirm the total fits. If you get fewer than eight weeks, compress Weeks 3–5 first; never compress Week 8\.

| Week | Chan Xing Szen | Ng Yong Vay | Ng Zhi Xuan | Joint milestone |
| ----- | ----- | ----- | ----- | ----- |
| **1** | Set up GitHub repo (1.10); download and verify both datasets; write ground-truth parser | Draft SMART objectives; freeze the three interface contracts | Draft the two comparison tables; collect 15+ academic sources | **Kick-off: lock objectives, contracts and signed Group Contract Form** |
| **2** | Tasks 1.3–1.5 (ingestion, denoising, enhancement) | Tasks 2.1–2.2 (registration \+ residual metric) | Task 3.1 (region descriptors) on synthetic blobs | Contracts frozen; everyone coding against stubs |
| **3** | Tasks 1.6–1.7 (binarisation, calibration); 1.11 morphology utility | Task 2.3 (differencing) | Task 3.2 (two-stage classifier v1) | **First end-to-end run on one image pair** |
| **4** | Tasks 1.4–1.6 benchmark experiments → Objective 2 evidence | Task 2.4 (blob extraction); begin integration spine | Tasks 3.5–3.6 (dashboard v1) | Preprocessing study complete |
| **5** | Task 1.8 (rectification for HRIPCB) | Task 2.5 (integration spine complete) | Tasks 3.3–3.4 (measurement \+ verdict) | **Integration checkpoint — full pipeline runs on the DeepPCB test set** |
| **6** | Task 1.9 unit tests; write 1.1, 1.3, 2.1, both tables | Tune registration to hit Objective 1; write 1.2, 2.2, 3.1 | Task 3.7 evaluation harness; run all metrics; export metrics CSV | **Objective 1 target (F1 ≥ 0.80) verified or formally revised** |
| **7** | Task 1.12 result figures; write 3.2, 5.2 | Write 5.1; assemble master document; code review backlog cleared | Write 4.1, 4.2; compile APA references | **Full draft circulated for internal review** |
| **8** | Final UK-English proofread pass; own AI Disclosure Form | Final integration test; slides; ZIP the source; **submit** | Slides; final reference check; own AI Disclosure Form | **Buffer \+ submission. Nothing new is built this week.** |

## 

## **5\. Submission checklist (leader only submits)**

* \[ \] **Part 1 — Documentation:** one editable file (.docx) **and** one PDF. Do **not** upload the originality report.  
* \[ \] **Part 2 — Single ZIP** containing source code and presentation slides. No dataset files inside — cite them in Section 3.2 instead.  
* \[ \] Appendix A — **Group Contract Form**, all three signed and dated, roles filled in (draft below).  
* \[ \] Appendix B — **AI Usage Disclosure Form**, one per member, individually completed and signed.  
* \[ \] Chan Xing Szen and Ng Zhi Xuan click **"Mark as done"** in Google Classroom.  
* \[ \] Cover page updated: replace "Project Title not Assignment Topic here" with the real title, and fill each member's "Algorithm/Module In-Charged".

**Cover page module labels to use:**

* (a) NG YONG VAY — *Image Registration & Defect Localisation Module*  
* (b) CHAN XING SZEN — *Preprocessing & Calibration Module*  
* (c) NG ZHI XUAN — *Defect Classification & Analysis Dashboard Module*

## 

## **6\. Appendix A — Group Contract Form, roles column (draft)**

| Student Name and ID | Role(s)/Task(s) Assigned |
| ----- | ----- |
| NG YONG VAY | Team Leader. Image registration and defect localisation module (feature-based alignment, registration quality metric, golden-template differencing, blob extraction). System integration and code review. Documentation: Section 1.2, Section 2.2, Section 3.1, Section 5.1. Final submission. |
| CHAN XING SZEN | Data acquisition, preprocessing and calibration module (dataset preparation, ground-truth parsing, noise removal, contrast enhancement, binarisation, spatial calibration, geometric rectification, morphological processing, unit testing). Repository setup and result figure generation. Documentation: Sections 1.1, 1.3, 2.1, Tables 2.1 and 2.2, Section 3.2, Section 5.2. Final UK-English proofreading. |
| NG ZHI XUAN | Defect classification, measurement and analysis dashboard module (region descriptors, two-stage rule-based classifier, physical measurement, pass/fail logic, Streamlit dashboard, evaluation harness). Documentation: Section 2.3, Section 3.3, Chapter 4, References. |

## 

## **7\. Team governance**

**Weekly checkpoint — 30 minutes, same slot every week.** Three questions each: what landed, what is blocked, what is next. Leader records it in a shared log. That log is your evidence if the Free-Rider Policy is ever invoked.

**Definition of done for any module task:** code merged to `main`, runs from the orchestrator without manual steps, has at least one test or documented sample input/output, and the corresponding report paragraph is drafted. Code that only runs on one person's laptop is not done.

**Free-rider protection.** The spec explicitly points to the TAR UMT Free-Rider Policy. Commit history plus the weekly log is objective evidence of contribution — this is the main reason for insisting everyone works in the shared GitHub repo rather than emailing files around.

**AI use.** Green-lit by the course for brainstorming, structuring and coding support, but **each member files their own AI Disclosure Form** listing tools, prompts and verification steps. Fill it in as you go; reconstructing it in Week 8 is painful and tends to be vague, which reads badly.

## **8\. If the group later decides to aim for Excellent**

You chose the solid-Good scope, which is the right call for a first plan. For reference, the spec is explicit that an Excellent grade requires evident extra effort. The three cheapest additions to bolt on later, in order of effort:

1. **Automated PDF report export** — Zhi Xuan, roughly two days on top of the existing dashboard.  
2. **Batch/folder ingestion** — Zhi Xuan, roughly one day; Module 1 already handles folder loading, so this is mostly a dashboard change.  
3. **Video/real-time frame analysis** — Yong Vay, roughly four days; the largest of the three.

Note the extras land mainly on Zhi Xuan and Yong Vay, because Xing Szen absorbed the rebalancing in Section 2A. If the group upgrades to the Excellent scope, re-run the balance table before committing.

Decide by the end of Week 5\. After that, adding scope threatens the Week 8 buffer.

## 

## **References for this plan**

Huang, W., & Wei, P. (2020). HRIPCB: A challenging dataset for PCB defects detection and classification. *The Journal of Engineering, 2020*(13), 303–309. https://doi.org/10.1049/joe.2019.1183

Tang, S., He, F., Huang, X., & Yang, J. (2019). *Online PCB defect detector on a new PCB defect dataset*. https://github.com/tangsanli5201/DeepPCB

