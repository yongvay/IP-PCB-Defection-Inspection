# ![][image1] 

# **Automated PCB Defect Inspection System: A Golden-Template Approach to Bare-Board Defect Localisation and Classification**

**BMDS2133 Image Processing**  
**Assignment Documentation**

**Semester 202605**

Group Members	:   
(a) NG YONG VAY		Image Registration & Defect Localisation Module  
(b) CHAN XING SZEN		Preprocessing & Calibration Module  
(c) NG ZHI XUAN			Defect Classification & Analysis Dashboard Module

Programme		: Bachelor in Data Science (Honours)  
Tutor			: Assoc. Prof Ts. Dr Tan Chi Wee  
Assignment Mode	: Mode B — Innovative Solution Development

# **Table of Contents** {#table-of-contents}

**[Table of Contents	1](#table-of-contents)**

[**Introduction	2**](#introduction)

[1.1 Problem Background	2](#1.1-problem-background)

[1.2 Objectives/Aims	2](#1.2-objectives/aims)

[1.3 Motivation	2](#1.3-motivation)

[**Literature Review	3**](#literature-review)

[2.1 Image Preprocessing and Enhancement	3](#2.1-main-concept-/-algorithm-1)

[2.2 Image Registration and Template Matching	3](#2.2-main-concept-/-algorithm-2)

[2.3 Morphological and Region-Based Classification	3](#2.x-main-concept-/-algorithm-3---add-more-if-needed)

[2.4 Analysis of the Existing Algorithms (Comparison)	3](#2.3-analysis-of-the-existing-algorithm-\(comparison\))

[**Methodology	4**](#methodology)

[3.1 System flowchart	4](#3.1-system-flowchart)

[3.2 Description of dataset (if Applicable)	5](#3.2-description-of-dataset-\(if-applicable\))

[3.3 Applications of the algorithm(s)	5](#3.3-applications-of-the-algorithm\(s\))

[**Result & Discussion	6**](#result-&-discussion)

[4.1 Experimental Results	6](#4.1-experimental-results)

[4.2 Discussion and Critical Interpretation	6](#4.2-discussion-and-critical-interpretation)

[**Conclusion	6**](#conclusion)

[5.1 Achievements	6](#5.1-achievements)

[5.2 Limitations and Future Works	7](#5.2-limitations-and-future-works)

[**References	7**](#references)

[**Appendix A	8**](#appendix-a)

[**Appendix B	9**](#appendix-b)

# 

# 

# **Introduction** {#introduction}

## 1.1	Problem Background {#1.1-problem-background}

> **\[TO WRITE — CHAN XING SZEN\]** Delete this box once written.
>
> Keywords in the project title needing their own defined paragraph: *printed
> circuit board* (what a bare board is, and why bare-board inspection happens
> before component placement); *defect* (the six standard classes and what each
> physically is); *golden template* (inspection by comparison against a
> known-good reference, as distinct from inspection by learned appearance);
> *automated optical inspection* (where this sits on a manufacturing line and
> what it replaces).
>
> Available to cite: Tang et al. (2019) for the six-class taxonomy and imaging
> conditions; Huang and Wei (2020) for the difficulty of real-world board
> orientation.


*Instruction to Student: Provide a clear narrative introduction to your project and its context. Ensure that every keyword in your project title is defined and explained in its own dedicated paragraph. Do not use bullet points or numbered lists here. All claims, statistics, and background facts must be backed by academic citations in APA format. Write strictly in UK English and avoid first-person pronouns (e.g., do not use "I", "we", "my").*

* *Citation needed for the problem that you mentioned*

## 1.2	Objectives/Aims {#1.2-objectives/aims}

The project pursues three objectives. Each follows the SMART framework, each is owned by a different member, and each is evaluated directly in Chapter 4 and assessed in Section 5.1.

**Objective 1:** To develop a golden-template-based inspection pipeline that localises and classifies the six standard bare-board defect classes on the DeepPCB test set, achieving a mean F1-score of at least 0.80 at an intersection over union threshold of 0.5, by the end of Week 6\.

**Objective 2:** To determine, through controlled experiments, the best-performing combination of noise-removal filter and binarisation method for printed circuit board imagery, by benchmarking at least three filters and two thresholding techniques against detection F1-score, by the end of Week 4\.

**Objective 3:** To deliver an interactive inspection dashboard that processes a template–test image pair and returns an annotated defect report with per-defect physical measurements in millimetres in under three seconds per board, by the end of Week 7\.

Each objective is measurable from artefacts the system already produces. Objective 1 is measured by the evaluation harness against the annotated ground truth. Objective 2 is measured by re-running that harness with one preprocessing setting changed at a time, so that filter choice is scored by its effect on detection quality rather than by visual impression. Objective 3 is measured by the wall-clock runtime recorded on every inspection report.

## 1.3	Motivation {#1.3-motivation}

> **\[TO WRITE — CHAN XING SZEN\]** Delete this box once written.
>
> The two figures below are supplied by the template and are the required
> anchors. SDG 9 links to resilient industrial infrastructure and the upgrading
> of manufacturing capability through quality assurance. Malaysia MADANI links
> to the national electronics manufacturing sector, particularly the Penang
> cluster.
>
> A defensible commercialisation angle is cost: the method uses classical image
> processing only, so it requires no annotated training corpus, no graphics
> accelerator, and no retraining when a new board design enters production — a
> new reference image is sufficient.


*Instruction to Student: Explain why this work matters. Discuss the potential commercialisation value or the tangible social impact of your project. You must explicitly link your project to broader societal frameworks, such as the **United Nations Sustainable Development Goals (SDGs)** or the **Malaysia MADANI** concept.* 

![][image2]  
Figure 1.1 SDG9 Industry Innovation and Infrastructure (United Nations, 2015\)

![][image3]  
Figure 1.2 Malaysia Madani (Jabatan Penerangan Malaysia, 2023\)

# **Literature Review** {#literature-review}

## 2.1	Image Preprocessing and Enhancement {#2.1-main-concept-/-algorithm-1}

> **\[TO WRITE — CHAN XING SZEN\]** Delete this box once written.
>
> Scope to cover, matching what Module 1 implements and benchmarks: noise
> removal by Gaussian, median and bilateral filtering, and why edge preservation
> matters when the features of interest are trace boundaries; contrast
> enhancement by global histogram equalisation, CLAHE and linear contrast
> stretching; binarisation by global Otsu thresholding (Otsu, 1979) against
> adaptive mean and adaptive Gaussian thresholding, and the bimodal-histogram
> assumption that makes Otsu appropriate for a bare board.

## 2.2	Image Registration and Template Matching {#2.2-main-concept-/-algorithm-2}

The golden-template method compares a board under inspection against a defect-free reference. That comparison is only meaningful when the two images occupy the same coordinate frame, so that every pixel of copper in the reference falls on the corresponding pixel of copper in the test image. A misalignment of even two or three pixels produces a bright outline along every trace in the difference image, and a connected-component stage then reports each outline as a defect. Registration is therefore not a preparatory convenience but the algorithm on which the validity of the entire comparison rests.

Zitová and Flusser (2003) divide registration methods into area-based and feature-based families. Area-based methods compare intensity patterns over windows and are effective when the images differ by a simple transformation and carry sufficient texture. Feature-based methods instead detect a sparse set of distinctive points, describe the neighbourhood of each, match descriptors between images, and estimate the transformation from the matched pairs. The feature-based family tolerates larger geometric differences and partial occlusion, which is why it is adopted here as the primary strategy.

Feature detection in this system uses ORB (Rublee et al., 2011), with SIFT (Lowe, 2004) available as an alternative. SIFT detects extrema in a difference-of-Gaussians scale space and describes each keypoint by a histogram of gradient orientations, producing descriptors invariant to scale and rotation and robust to moderate illumination change. ORB combines the FAST corner detector with a rotation-aware BRIEF descriptor, yielding a binary descriptor matched by Hamming distance. The practical distinction is cost: ORB is substantially faster and free of the patent constraints that historically limited SIFT, at some loss of robustness under scale change. Since a board presented to a fixed inspection camera varies little in scale, ORB is the appropriate default and SIFT is retained as a fallback for difficult pairs.

Descriptor matching alone is insufficient, because repetitive structure guarantees false matches. A bare printed circuit board approaches a worst case for descriptor matching: it consists largely of parallel traces and regularly spaced pads, so many keypoints have near-identical neighbourhoods and match confidently to the wrong location. The transformation is therefore estimated with RANSAC (Fischler & Bolles, 1981), which repeatedly fits a candidate homography to a minimal random subset of matches and retains the model supported by the largest set of inliers. This tolerates a high proportion of incorrect correspondences, which is precisely the regime this imagery produces.

Where descriptor matching fails outright, phase correlation provides a fallback. Reddy and Chatterji (1996) describe the technique: the normalised cross-power spectrum of two images has an inverse Fourier transform sharply peaked at the translation between them. The method uses the whole image at once rather than a sparse set of points, so it is unaffected by the scarcity of distinctive keypoints, and it recovers translation robustly. It cannot recover rotation in its basic form, which is exactly why it serves as the fallback rather than the primary method.

Two design decisions in this system follow from the literature but were settled empirically rather than assumed.

The first concerns whether to register at all. The registration literature generally treats alignment as an unconditional preprocessing step, but the DeepPCB pairs were aligned by their authors before publication. Fitting a homography to an already-aligned pair cannot improve the alignment and can degrade it, because the estimate is then fitted to noise. Measurement confirmed the concern: applying the feature-based transform unconditionally cost approximately 0.11 of F1. The system therefore treats the identity transform as an explicit candidate, scores every candidate by the proportion of pixels that still disagree after warping, and adopts whichever scores best. Registration is thus conditional, and the decision is evidence-driven at inspection time rather than fixed at design time.

The second concerns reporting alignment quality. Registration can fail silently: a poor homography still produces an output image, and the failure surfaces only as an implausible number of defects. The system therefore computes the mean reprojection error of the inlier correspondences and carries it on every inspection report, so that a badly registered pair is flagged rather than quietly generating false detections. A threshold of three pixels is used, which at the dataset scale of roughly 48 pixels per millimetre corresponds to about 0.06 mm — tighter than the smallest defect the system is expected to resolve, and loose enough not to reject sound alignments over sub-pixel noise.

## 2.3	Morphological and Region-Based Classification {#2.x-main-concept-/-algorithm-3---add-more-if-needed}

> **\[TO WRITE — NG ZHI XUAN\]** Delete this box once written.
>
> Scope to cover, matching what Module 3 implements: morphological opening and
> closing, and the geometric reason opening separates registration jitter from
> genuine defects (jitter along a trace edge is long and one or two pixels wide,
> whereas a real defect is compact); connected-component labelling and contour
> extraction as the basis for treating each difference region as a measurable
> object; region descriptors including area, perimeter, aspect ratio, extent,
> solidity and Hu's moment invariants (Hu, 1962); and rule-based against learned
> classification, with the argument that a rule-based scheme is defensible when
> the class definitions are geometric rather than appearance-based.

## 2.4	Analysis of the Existing Algorithms (Comparison) {#2.3-analysis-of-the-existing-algorithm-(comparison)}

> **\[TO WRITE — CHAN XING SZEN\]** Delete this box once written. Both tables
> below are owned by Module 1 and are the evidence for Objective 2. Populate the
> "Example of Previous Work" column with citations from the library databases.
>
> **Numbering note:** a third algorithm section is used, so this comparison
> section is renumbered from 2.3 to **2.4** and the chapter runs 2.1 → 2.2 →
> 2.3 → 2.4.



Table 2.1 Comparison of Contemporary Approaches in Image Segmentation

| Algorithm / System | Example of Previous Work *(insert as citation)* | Advantages | Limitations | Remarks |
| :---- | :---- | :---- | :---- | :---- |
| **Algorithm 1:** \[Name\] |  |  |  |  |
| **Algorithm 2:** \[Name\] |  |  |  |  |
| **Algorithm 3:** \[Name\] |  |  |  |  |

Table 2.2 Comparison of Contemporary Approaches in Contrast Enhancement

| Algorithm / System | Example of Previous Work *(insert as citation)* | Advantages | Limitations | Remarks |
| :---- | :---- | :---- | :---- | :---- |
| **Algorithm 1:** \[Name\] |  |  |  |  |
| **Algorithm 2:** \[Name\] |  |  |  |  |
| **Algorithm 3:** \[Name\] |  |  |  |  |

 

# 

# **Methodology** {#methodology}

## 3.1	System flowchart {#3.1-system-flowchart}

As illustrated in Figure 3.1, the system executes as a linear pipeline of three modules joined by two fixed data contracts. Each module is independently replaceable: any stage may be rewritten without the others being modified, provided the shape of the structure crossing the boundary is preserved.

Three properties of the architecture are worth stating explicitly, because each was a deliberate decision rather than an accident of implementation.

The classification problem is halved before any shape is measured. The difference between the reference and the test image is computed as a *signed* quantity rather than an absolute one. Copper present in the reference but absent from the test image indicates copper removed, which restricts the defect to an open circuit, a mouse bite or a pin hole. Copper present in the test image but absent from the reference indicates copper added, which restricts it to a short, a spur or spurious copper. Two bitwise operations therefore resolve half of the six-way classification, and the shape descriptors need only separate three classes within a group rather than six overall.

Morphological cleanup is placed between differencing and blob extraction, and belongs to Module 1\. Its position in the sequence is an integration decision, but its implementation and tuning remain with the owner of the preprocessing module. It is exposed as a single pure function with the same array shape in and out, so the orchestrator calls it without either module depending on the internals of the other.

The module boundaries are validated rather than merely documented. The structure passed from Module 1 to Module 2 checks its own invariants — matching image shapes, an unsigned 8-bit type, strictly binary values and a positive calibration factor — and raises an error at the boundary. A fault therefore surfaces where it originates rather than as a confusing failure several stages downstream.

*![][image4]*  
**Figure 3.1:** System Architecture and Process Flowchart.

> **\[ACTION — NG YONG VAY\]** Replace the placeholder above with a
> high-resolution export. The Mermaid source below reproduces the pipeline as
> built; render it at <https://mermaid.live>, export as PNG at 2x scale, and
> delete this box.
>
> ~~~
> flowchart TD
>     A["Template image (defect-free reference)"] --> B
>     A2["Test image (board under inspection)"] --> B
>     subgraph M1["MODULE 1 - Preprocessing and Calibration"]
>         B["Image ingestion, greyscale conversion"] --> C["Noise removal: Gaussian / median / bilateral"]
>         C --> D["Contrast enhancement: histogram equalisation / CLAHE"]
>         D --> E["Binarisation: Otsu / adaptive"]
>         E --> F["Spatial calibration: derive mm per pixel"]
>     end
>     F --> G{"PreprocessResult: template_bin, test_bin, mm_per_px"}
>     subgraph M2["MODULE 2 - Registration and Localisation"]
>         G --> H["Feature detection: ORB / SIFT keypoints"]
>         H --> I["Descriptor matching + RANSAC homography"]
>         I --> J{"Alignment residual under 3 px?"}
>         J -- no --> K["Phase correlation fallback"]
>         J -- yes --> L
>         K --> L["Score candidates: identity vs warp"]
>         L --> M["Signed differencing: removed and added"]
>         M --> N["Morphological cleanup: opening then closing"]
>         N --> O["Connected components: boxes, contours, centroids"]
>     end
>     O --> P{"LocalisationResult: blobs with polarity, align_residual"}
>     subgraph M3["MODULE 3 - Classification and Analysis"]
>         P --> Q["Stage 1: polarity, copper added or removed"]
>         Q --> R["Region descriptors: area, aspect ratio, solidity, Hu moments"]
>         R --> S["Stage 2: rule-based, three classes per group"]
>         S --> T["Physical measurement: pixels to mm2"]
>         T --> U["Board verdict against defect tolerance"]
>     end
>     U --> V["InspectionReport: defects, verdict, runtime"]
>     V --> W["Streamlit dashboard: overlay, table, PDF export"]
>     V --> X["Evaluation harness: IoU matching, precision / recall / F1"]
> ~~~

## 3.2	Description of dataset (if Applicable) {#3.2-description-of-dataset-(if-applicable)}

*Instruction to Student: Clearly state the source of your data (provide the exact URL link if obtained online, or detail the collection method if self-collected). Outline the data structures or provide a comprehensive Data Dictionary. Include a sample image or snippet of the raw dataset.*

> **\[TO WRITE — CHAN XING SZEN\]** The facts below are verified against the
> copy of the dataset in the repository. Write them up as prose, add a sample
> image pair with ground-truth boxes overlaid, and delete this box.

**Dataset source.** Primary: DeepPCB (Tang et al., 2019), <https://github.com/tangsanli5201/DeepPCB>. Secondary: HRIPCB (Huang & Wei, 2020), <https://doi.org/10.1049/joe.2019.1183>.

**Verified structure.**

| Property | Value |
| :---- | :---- |
| Image pairs | 1,500 |
| Annotated defects | 10,013 |
| Split | 999 training/validation pairs, 499 test pairs |
| Image size | 640 × 640 pixels, greyscale |
| Scale | approximately 48 pixels per millimetre |
| Defects per image | 3 to 12 |
| Annotation format | one defect per line: x1 y1 x2 y2 type |

**Data dictionary — annotation type codes.**

| Code | Class | Polarity |
| :---- | :---- | :---- |
| 1 | Open circuit | Copper removed |
| 2 | Short | Copper added |
| 3 | Mouse bite | Copper removed |
| 4 | Spur | Copper added |
| 5 | Spurious copper | Copper added |
| 6 | Pin hole | Copper removed |

**Two properties of this dataset must be stated here, because both materially affect Chapter 4\.**

Copper renders dark against a bright substrate. Approximately 86% of every image is bright substrate, and a copper-removed defect makes its region brighter rather than darker. Assuming the intuitive opposite convention inverts every polarity label, which inverts the first stage of the classifier while leaving the bounding boxes looking entirely correct.

Annotation boxes are drawn with a margin. Ground-truth boxes mark a region around each defect rather than the changed pixels themselves, with a margin of approximately ten pixels measured consistently across the sample examined. Scoring tight predicted boxes against padded ground-truth boxes yields a recall of 0.01 at an IoU threshold of 0.5; applying the same convention to the predictions yields 0.71 from identical detections. The system is no better in the second case — the comparison is finally like for like. This must appear as a stated methodological decision with both figures, not as an unexplained constant in the source code.

*Sample image: insert a template–test pair with ground-truth boxes overlaid.*

## 3.3	Applications of the algorithm(s) {#3.3-applications-of-the-algorithm(s)}

*Instruction to Student: Describe exactly how the selected algorithm(s) or technique(s) are practically applied, coded, or integrated within your specific system implementation.*

> **\[TO WRITE — NG ZHI XUAN\]** This section compiles pseudocode from all
> three modules; each owner supplies their own. A verified worked example is
> reproduced below. Delete this box once written.

*![][image5]*

Sample input and output, produced directly from the orchestrator:

~~~
python -m src.pipeline data/DeepPCB/PCBData/group00041/00041/00041000_temp.jpg \
                       data/DeepPCB/PCBData/group00041/00041/00041000_test.jpg

Verdict:   FAIL
Defects:   9
Residual:  0.007 px
Runtime:   0.051 s
  #0   mouse_bite       (437, 35, 9, 7)     0.0204 mm^2
  #1   mouse_bite       (502, 38, 14, 8)    0.0356 mm^2
  #2   open_circuit     (550, 271, 29, 33)  0.1727 mm^2
  #3   mouse_bite       (260, 351, 9, 10)   0.0286 mm^2
  #4   spur             (161, 159, 12, 7)   0.0291 mm^2
  #5   spur             (345, 263, 10, 9)   0.0221 mm^2
  #6   short            (468, 317, 16, 65)  0.1645 mm^2
  #7   spurious_copper  (231, 324, 12, 12)  0.0438 mm^2
  #8   spur             (99, 479, 19, 9)    0.0590 mm^2
~~~

# **Result & Discussion** {#result-&-discussion}

## 4.1	Experimental Results {#4.1-experimental-results}

> **\[TO WRITE — NG ZHI XUAN\]** The measurements below were taken on the
> held-out test split and are supplied as the seed for this section. They are
> not the finished analysis: task 3.7 replaces this with per-class precision and
> recall, a full confusion matrix, runtime benchmarking, and a metrics CSV for
> Chan Xing Szen to chart. Delete this box once written.
>
> Figures still required: an F1 comparison across preprocessing settings
> (Objective 2 evidence, from Module 1's benchmark), a defect-count distribution
> per class, and a runtime distribution against the three-second budget.

Table 4.1 Detection performance over 150 held-out test pairs

| Scoring basis | IoU threshold | Precision | Recall | F1 |
| :---- | :---- | :---- | :---- | :---- |
| Localisation only | 0.33 | 0.904 | 0.779 | **0.837** |
| Localisation only | 0.50 | 0.891 | 0.768 | **0.825** |
| Localisation and class | 0.33 | 0.437 | 0.376 | **0.404** |
| Localisation and class | 0.50 | 0.433 | 0.373 | **0.401** |

Mean runtime: 0.058 s per board.

Table 4.2 Confusion matrix over 120 held-out test pairs at an IoU threshold of 0.5. Rows are ground truth, columns are predicted, and counts are matched detections.

| Truth / Predicted | Open circuit | Mouse bite | Pin hole | Short | Spur | Spurious copper | Recall |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Open circuit** | **50** | 59 | 14 | 0 | 0 | 0 | 41% |
| **Mouse bite** | 20 | **43** | 9 | 0 | 0 | 0 | 60% |
| **Pin hole** | 19 | 34 | **22** | 0 | 0 | 0 | 29% |
| **Short** | 0 | 0 | 0 | **61** | 32 | 4 | 63% |
| **Spur** | 0 | 0 | 0 | 10 | **52** | 21 | 63% |
| **Spurious copper** | 0 | 0 | 0 | 17 | 53 | **28** | 29% |

Stage-one polarity classification was correct on 548 of 548 matched detections, or 100%.

## 4.2	Discussion and Critical Interpretation {#4.2-discussion-and-critical-interpretation}

> **\[TO WRITE — NG ZHI XUAN\]** Four observations the data above supports,
> offered as a starting structure. Delete this box once written.
>
> The zero block in Table 4.2 is the central result. No copper-removed defect
> was ever predicted as a copper-added class, or the reverse. Signed
> differencing does not approximate the polarity split; it determines it. This
> is the strongest evidence in the report that the architectural decision in
> Section 3.1 was correct.
>
> All classification error is confined within a polarity group. The gap between
> an F1 of 0.825 on localisation and 0.401 with class labels is therefore
> attributable entirely to the stage-two descriptor rules, and not to detection,
> registration or differencing.
>
> The stage-two thresholds are unfitted. Aspect ratio and solidity are split at
> hand-chosen constants. Pin hole and spurious copper, the two classes at 29%
> recall, are both the compact case within their group, which points at the
> solidity threshold rather than the descriptor set.
>
> Precision exceeds recall consistently. The system misses defects more often
> than it invents them, which for an inspection application is the less
> desirable of the two failure modes and should be discussed as such.



# **Conclusion** {#conclusion}

## 5.1	Achievements {#5.1-achievements}

*Instruction to Student: Synthesise your findings and explicitly state whether your project successfully fulfilled the SMART objectives outlined in Section 1.2. Declare which algorithm proved to be the best based on your experimental data.*

> **\[REVISE BEFORE SUBMISSION\]** This section reports each objective against
> the measurements available at the time of writing. It requires a final pass
> once Objective 2's benchmark and the task 3.7 harness are complete.

**Objective 1 Status: Partially Fulfilled.** Objective 1 set a mean F1-score of at least 0.80 at an IoU threshold of 0.5 for a pipeline that both localises and classifies the six defect classes. Measured on the held-out test split, the system achieves an F1 of 0.825 on localisation, exceeding the target. When class labels are also required to match, F1 falls to 0.401. The objective is therefore met on the localisation component and missed on the classification component, and is reported as partially fulfilled rather than fulfilled. The measurements locate the shortfall precisely: stage-one classification, which assigns copper-added or copper-removed polarity from the sign of the template–test difference, was correct on all 548 matched detections, and every classification error lies within a polarity group rather than across groups. The deficit is thus wholly attributable to the stage-two descriptor rules, whose thresholds on aspect ratio and solidity remain at hand-chosen values rather than values fitted on the training split. This is a tractable problem with a clear route to improvement, not a failure of the approach.

**Objective 2 Status: Not Yet Assessed.** Objective 2 requires at least three noise-removal filters and two binarisation methods to be benchmarked against detection F1-score. The evaluation harness needed to produce that comparison is in place and the pipeline accepts the relevant parameters, but the benchmark itself has not yet been run. No claim is made in this report about the best-performing combination, and the objective is reported as outstanding.

**Objective 3 Status: Fulfilled.** Objective 3 set a budget of three seconds per board for an interactive dashboard returning an annotated defect report with physical measurements in millimetres. Mean measured runtime is 0.058 s per board, approximately fifty times inside the budget. The dashboard accepts a template–test pair, displays the annotated overlay with defects colour-coded by polarity, lists each defect with its class and area in square millimetres, and exports the result as a PDF report. The PDF export was scheduled in the project plan as an optional extension and has been delivered ahead of that schedule.

**On the best-performing algorithm.** The template asks which algorithm proved best. On the evidence available, the decisive choice was not a filter or a threshold but the decision to compute the template–test difference as a signed rather than an absolute quantity. That single decision resolves half of the six-way classification exactly, at a cost of two bitwise operations, and it is the reason the polarity split is perfect where the geometric rules built on top of it are not. A secondary finding of comparable weight is that morphological opening with a 5 × 5 elliptical element raises precision from 0.10 to 0.93 on a fixed sample, because binarisation jitter along a trace edge is long and one pixel wide whereas a genuine defect is compact — a geometric distinction rather than a statistical one.

***Project Code Repository:** \[Insert the GitHub repository URL here\]*

## 5.2 Limitations and Future Works {#5.2-limitations-and-future-works}

> **\[TO WRITE — CHAN XING SZEN\]** Known limitations already established and
> available to write up. Delete this box once written.
>
> Stage-two classification thresholds are unfitted. Aspect ratio and solidity
> thresholds are hand-chosen constants, giving within-group recall between 29%
> and 63%. Fitting them on the training split is the single highest-value
> improvement available.
>
> HRIPCB has not been evaluated. The secondary dataset contains no template
> images, so the golden-template assumption does not hold for it without a
> reference board being synthesised or selected. This limits the evidence
> available for the geometric rectification requirement.
>
> Registration is verified on synthetic transformations. Rotation and shift
> correction are demonstrated on artificially transformed DeepPCB boards, which
> is weaker evidence than genuinely misoriented imagery.
>
> Spatial calibration is a quoted constant. The scale factor of 48 pixels per
> millimetre comes from the dataset documentation rather than being derived from
> a measured board feature, so every measurement in square millimetres inherits
> that assumption.
>
> Confidence scores are placeholders. The rule-based classifier produces a
> decision but no graded score, so the confidence field carries a fixed value
> and must not be interpreted as a probability.



# **References** {#references}

*Instruction to Student: Compile an exhaustive bibliography of all scholarly material referenced within this assignment. You must strictly adhere to the APA referencing format, ensuring each entry includes the author, publication year, title, and publisher or URL. Every in-text citation must have a matching full entry.*

* *Specify the Digital Object Identifier (DOI) for your dataset where available.*   
* *Avoid websites/forums/researchgate/arxiv if possible.*  
* *Numbered lists are not permitted in this section.*  
* *Organise all bibliographic entries in alphabetical order.*  
* *Employ academic search engines or university library databases to ensure the precision and standardisation of your reference metadata.*

> **\[TO COMPILE — NG ZHI XUAN\]** The entries below are those cited in the
> sections drafted so far. Each must be verified against the TAR UMT library
> databases before submission, and the remaining sections will add to this list.
> Keep the list alphabetical and unnumbered. Delete this box once compiled.

Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. *Communications of the ACM, 24*(6), 381–395.

Hu, M.-K. (1962). Visual pattern recognition by moment invariants. *IRE Transactions on Information Theory, 8*(2), 179–187.

Huang, W., & Wei, P. (2020). HRIPCB: A challenging dataset for PCB defects detection and classification. *The Journal of Engineering, 2020*(13), 303–309. https://doi.org/10.1049/joe.2019.1183

Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision, 60*(2), 91–110.

Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics, 9*(1), 62–66.

Reddy, B. S., & Chatterji, B. N. (1996). An FFT-based technique for translation, rotation and scale-invariant image registration. *IEEE Transactions on Image Processing, 5*(8), 1266–1271.

Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. *2011 International Conference on Computer Vision*, 2564–2571.

Tang, S., He, F., Huang, X., & Yang, J. (2019). *Online PCB defect detector on a new PCB defect dataset*. https://github.com/tangsanli5201/DeepPCB

Zitová, B., & Flusser, J. (2003). Image registration methods: A survey. *Image and Vision Computing, 21*(11), 977–1000.

>   
> \_

> # Appendix A {#appendix-a}

**TUNKU ABDUL RAHMAN UNIVERSITY OF MANAGEMENT AND TECHNOLOGY**  
**Group Contract Form**

Faculty					: Faculty of Computing and Information Technology (FOCS)  
Programme				: Bachelor in Data Science (Honours)  
Year of Study				: \[to complete\]  
Semester				: 202605  
Course Title				: Image Processing  
Course Code				: BMDS2133  
Group Assignment / Project Title	: Automated PCB Defect Inspection System  
Name of Tutor				: Assoc. Prof Ts. Dr Tan Chi Wee  
Tutorial Group				: \[to complete\]

| Student Name and ID | Role(s)/Task(s) Assigned | Signature and Date |
| :---: | :---- | :---: |
| NG YONG VAY<br>\[ID\] | Team Leader. Image registration and defect localisation module: feature-based alignment, registration quality metric, golden-template differencing, blob extraction. Repository setup, system integration and code review. Documentation: Section 1.2, Section 2.2, Section 3.1, Section 5.1, Appendix A. Final submission. |  |
| CHAN XING SZEN<br>\[ID\] | Data acquisition, preprocessing and calibration module: ground-truth parsing, noise removal, contrast enhancement, binarisation, spatial calibration, geometric rectification, morphological processing, unit testing. Result figure generation. Documentation: Sections 1.1, 1.3, 2.1, Tables 2.1 and 2.2, Section 3.2, Section 5.2. Final UK-English proofreading. |  |
| NG ZHI XUAN<br>\[ID\] | Defect classification, measurement and analysis dashboard module: region descriptors, two-stage rule-based classifier, physical measurement, pass/fail logic, Streamlit dashboard, PDF report export, evaluation harness. Documentation: Section 2.3, Section 3.3, Chapter 4, References. |  |

> **\[OUTSTANDING\]** Student ID numbers, year of study and tutorial group are
> not yet filled in, and the form requires three signatures with dates. This form
> carries marks simply for being complete. Delete this box once signed.

> # Appendix B {#appendix-b}

**AI Usage Disclosure Form**

> **\[ONE FORM PER MEMBER\]** Three copies are required, each individually
> completed and signed. This cannot be delegated to the leader. Completing it as
> work proceeds is markedly easier than reconstructing it at the end, and a
> vague reconstructed statement reads poorly. Delete this box once all three are
> attached.

**Student Name		:  **                                                                  	

**Programme		:  **                                                                  	  
**Course			: **                                                                   	  
**Assessment Title	: **                                                                  	  
	  
**1\.**  **AI Tool(s) Used (Check that all that apply)**

- [ ] None (I did not use any AI tools for this assignment)  
- [ ] ChatGPT (Version:             	)  
- [ ] Deepseek  
- [ ] Gemini  
- [ ] Other:                                	

**2\.**  **Nature of Assistance (Check all that apply)**

- [ ] **Brainstorming:** Generating ideas, topics, or outlines.  
- [ ] **Research:** Summarising long articles or finding concepts.  
- [ ] **Editing:** Checking grammar, spelling, or sentence structure.  
- [ ] **Coding/Math:** Debugging code or explaining a formula.  
- [ ] **Creation:** Generating images, data, or draft text.

**3\.**  **Disclosure Statement**

   
Provide a brief explanation of how AI tools were utilised in completing the task and describe the measures taken to ensure that the final submission reflects your own independent work.

   
**Example:**  
*“I used ChatGPT to structure my initial ideas into an outline. Subsequently, I composed the full paragraphs independently and cross-checked the referenced dates against my textbook to ensure accuracy.”*

   
**Your Statement:**

|   |
| :---- |

**4\.**  **Verification & Integrity**

- [ ] I have reviewed and verified the accuracy of all facts, data, and citations generated with the assistance of AI tools.  
- [ ] I have properly acknowledged and cited any AI-generated text, ideas, or content in accordance with my instructor’s requirements.  
- [ ] I confirm that the reasoning, interpretation, and analysis presented in the final submission are entirely my own.

 

   
**Student Signature	:                                                                         	 Date			:**                                                                         	  
   
   
 

   
AMM 5.3.2026/SMC 1.4.2026  


[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAP4AAABoCAIAAABey9pYAAAlX0lEQVR4Xu19CXgc1ZXu7b21eNHSXV1d1d2S5Q0bMAZDJoGZITAhJEDGBgzGm2zL2lpSt1otyZJsYwhDMryXzCSZfBkmGZK8L5lMPpKZbxgSkpchBPAKMVtIhiUMhISwG7Cspbeq+845Vd1VarWFbfCS1/V/l3ZTfeveW/f+99xzzj23xLgFC2UJVnzBgoXygEV9C2UKi/oWyhQW9S2UKSzqWyhTWNS3UKawqG+hTGFR30KZwqK+hTKFRX0LZQqL+hbKFBb1LZQpLOpbKFNY1LdQprCob6FMYVHfQpnCor6FMoVFfQtlCov6FsoUFvUtlCks6lsoU1jUt1CmsKhvoUxhUd9CmcKivoUyxZ8E9Se4ynkOU4arO7/wE2/9RrgC1+g/CxZOBH8C1E9nRid5Nr7jHhYaYoGELRitFju4YlHfwgfCGUj9DH7ksuNA7JxyQ9fXWaiXheJMjLp8Pb6GYSbHK8ReyKIi75Wp91qwcKw486iPhM7+4fUcC8VYsJ8Foqwxtn7B1TwL1ycu/LPtTEpWBLsxo0V9Cx8AZwj1dQarIPJz8Jl64TXOggNsdvuvl1z49HkX/Ney83ga8mSXX9LP5KQh9VW4qKg0LeiLpQFZOFacEdQnyiKz4Z/krd/3BNtZeLszEGfB5HNLlj+z5Lyfn3VhCnMp539sO6wGhtRXifSaEUzTByePBQvHgNNPfUVRxvlk5+B3mL+fhYdYcKtH7krsuP/YqA//ZoD+Y5y7fTe8Om5pQBaOFaeX+ko6o1Y0bWINXQ6hxxZpb1uyaTyDK8Dv/shnoH61kAC9f2ziSJZnnXO3VIS3s/oEk2O/e724AgsWjobTSX2VH5mAFoR6vLVdT5578ZPnLH5m4UWTqPfwl1+difrM2+aqv8YxbwuT+9nsQYf/M3YwiC3qWzgenFbqKxyIzqQ+u9j5PwvOAd4/dvZyzVD97eszUr8xXlszzPwd1358W470e694vFJfyYJVkMFpluNZWGfok2fQbKZUZDKjGYI7a1SdlllJoXJF7TJyYXE5nimUqd+ilUY5yarJAzNzMtCx1hT+T8r49ejIwIqHxYwVKlLwfzPKJDZe5amJyTQ0l37NUJuLHWLpI2hc0UX8NYcdoqQmaFTULHYM3pNNU4NGczhS9LygWmZUJY33UtU8O5rG7UW9T2AdxkoyY4WKoBCqBB/O1NWFT9qgMXf1qcLppD48NHQrk3uZ1PnbRYufPmvFI8uWa13xuxmlvmd288GFH3uyadHo7V+nfsu6gsnjoz6OFqpW0uJu3DTQUz8L7Jg7fzMREX81YaIqstYmbmfhwUL+ikAyC8QwMVkbRCasNZXZy0LbHPLQ6zgtstocKgAIwvwjTNrGwn2Q0xFK0Ex4fwD1WHCYScNGRZGYXbgFjB3kLk7dFPNvxb6V4dceJo+45daphU987qsPMYEyQAIBJPUcymgsTD33e+7xJ1ioE38SOpnUxbCH+1iwi/m3XNf2ZZrkNLezOa+w3tyHrsB63HY3gB0EVdukVhgjU05MR7A/TPPkFOJPk/piy7Pzlz1x1gVHPnfHiVEfboHBU9NKYFGUBh6H3yYmnTC6/jiIN2KoQWqQefbIbUwcsAVjWmb49AZ6zHkICohAFhwslOn072BSNwv0eeUWXe6a7gBpyuSkI5C0BbuZlGCB7mOk/sKPdAEvHYJREZMGmJyQlrbBUpqDCUncc83big0OxmwidObIZ9bdXigB11uhC7koRyGDXW5/cQxbl+Vjc+ffxuZ02AID2HJYYyUoIYFVBDG55A7mG6yQ2tO4oMEykfEEm6EEvSVSokJq1fcldeDyct/9LzIpaWptoQ+vpw4p6sZTgdNJfdQMgChS//FSvzLQ8sSSxf+9cPmR2//uxKjPtU2ANEfq04iak1fcSHlQ8BcEubuhn36NFbJ5gp3a0lFALpt21Kxk4W5HIAEkIEIbxaaoqVOkfpYzcUjjvU1M2PzxmamPt5K2UTEf5onWmARyVByE8m3BKFBc2+PQ8E6G2wJDTKb2yN1zFwxlQDPK8HTucFVkPZMGmRSF26Fqp3gtPmya1zauYbXthTZXR3Y2nr317p+8dcEn/8YR7HAIIw4xBtPJGexxSW3E2IxbbGVivltCsWppk5n61OasI9CptZaJSZqHCcoPXTQ4CfWqMz/3ScHppP6YwjfG78JlVOw6o6jP/Mn3MAvXhhAFpDigk15MFrKVoD4uYn1OgbgY2HGYcwcOMyU5XlXfTAF4hoQ7EerziXVd38y3ASZY3zuTnIW3wXfkd2Rw5YavFvJnJ8eEczbaZaPNH7tqFzTgMOpLegnQQs+8TrQMYMlSctqVQv7D2OsZXK8UnlO5q/7GimBfRXCwUhyCL2gHqBlHsG0G6kO5YJGA1oRrVKiNBXoPQbcIoPlEYbmDUauUN56W3ZhTR33UC2G1V9OTPAU6pSvUCqrerMYhWknbTw71tRg3sxwsoCT1E7T6w2ebW2wDGU5UO2ITWnBrGZIURR0gn99MffzI8L9Y+VlWBypyq12I2up709AwGfJ3kpCLMX87mI9amzQcL/UxiJVnmK+r0IZZ85LwlGz2FiyEroBUVvkRFOFkgsPDgx5Fk6Qb5m1Vw85JnvUGenGeQH5QZuqH3kPlD9s1kZ2ERVj/iVJVqPW9ST4BT5JJoRmtHNFpTf2OrVU4DMdM1FcUVtcGKxKTOkAls9d0Z/lEBSo8uCRCs21g4ZjExynDqaO+OqkeVrhTuMkm7nSAYurfVBm+7pUjJ6LrHyP1cxkly99DwaSqCnlATChF/UALbi9IvSizpRgKPJCboJIFYIRAXU445F5PYymFB3OmJkC7EOO2AOoe0JgLPjEATL3tyw8yYRuUhhJOGrip5U6zCXj81M/+7g3uDA8U2vDEs5NQe03jBuMpgonnf6/zHh4zm5v8wztQi/aYMSb2umB6i33IPFjK/D2zxKsML5PKxbO3VjXtLJSGKwBqUwm0JYJJt5xwCNdPKuiaOTz21rFQHy1yfydSXIRCYudfmoBp9Pl/+jnqaaikwbMPfGbtFwr5TxlOHfVZOF5Z1wem4aVVlz921mXw/LBWZtDw72PSB5b6wNfgyEtvcRzCvI+wMrLGC2u9mCxuCqIE9avk2P+6817mS2q6TU0TWLGpCl8zLgWgw8xrS2z/d1dgayF/EfW//t3fkGZPGq2/dxRryYylOfP1kUoN06nDKQ0Tz7SFKHu81B8bG0XtQkJtGz6BxCo9yBE0mfLUl6KzwzszykQqpflJcc5XydfC4oOamN74mEPsxqZKURK4mAe/qBym+pxGNFuRrNBmlM0wYQbhu1tI6FY+WOS+TW/TU8MtnuDGEtQnwQGoW9LL6tB4cDdAJwylc1gbrKdQoDOitaEP6sIQLX2VPkU4ddR34qj03iFf/syCxqebzgGZDMYNUR+67ANLfQnF0ktvIONr5PW4wyUNOGQcORit4qYgSlI/Op4bY3VokIFMgnV//wucCok6pXbm71b4uIk9ZoUnCwPq8LcUfprVsC6ln64B6q/TlFobrPj1se/+x7OFNhwv9XEJwhWpA7krDNl9K3M0jSaz0I1thdrBVjkCwtYkelOZtCfQx8KtRp5gEohIzlA9D0p9BfUTuHFc5ZXBNbiIYULbFF09sLTq5mncJvcwfyvdNwHKYQnqI7KgLNnlhCtCHoJAnyPSOYnrES4FFU2bvI030/UeSOOooZVUTU8WTh310XMc6Nk575r/Wbj0qaVLQW/8EKnvJoUHWRveBV0M2hSwCnrZiwOWKG4KogT1veF2dTz1wltcV83lbld4hImduDT7eg6+jCxzylsK+c1S/49vAikNFRmWMlekEz49kWEWadMcPqCFs8akN9xPbEMpd7zUd4sbwZAgVYHcR5GkHWZCqMfu7yEXJFUNzYh0VAnNVEMeCkfTtnZboYUOoffSG29VxzOmLTbskxztPcGjZtJ8MoWafTqdBkrC7U55LfWM7pyBNk8iwyfcweYS1CeJ8JErjD5xBiBPP0qlhrhd3MZ8MSb0FH6tDK7W+uSU4dRR3y51MbFnZ9Onnl+w4tnFizmq4B8a9UHqe30Ddnl1a+P1B+b/JR9/dwxl5KRdAEUoXtwURAnqV0pJ3F/MqV651RFGqa+74aR4RUMvEgK0edkwMQvUz6np6tAWXGF0t10C1WhKIIzBwgONuWCGQsrLWrPUj6FrUuihp5yWOJ7aAUq6JagCbIaYE0+r4eQsVITUD0VRGxFxlwBk7RQRSl7aCsmgGgt2T+jNMPBWhleG180ObpklN9c0tFTLN6JZrij47DkQ1WNeWP1MPq5x6v1KoZRzk6Q4LvX5RI4jdMU60J0Ka84gyoJChkAfNnhqe04q/j+hvgMVnh3/umLdQysaXpz/UVxSTpD6gNRhUNCFPk3w40iL0RfexqgAkPr2YF8hf4H6h8dhoUBFlhSbblCNPOJNHnEdfLqC8Hkjq13PTLapR2jFZV8xSf1QFJ7X1TDsIN98UUpT5vnLtNtJjoaiFY1dnuC1hYo8tde7gP1ojMagEObvb7rA9ODTqS/HMJTBTDVSOICXLATLVNIhDNJuwKav/J8DMOseeWZ0VqQTRlB38pKFmsHbs5Xi2hLU55knXuAsZMwTVtdaGdpSEcAGVwhbZoduYLNMPgY5fv7HdxxjHMeHgjOD+mLP84sXP7VkxYFzLobOHOejr6a415esnh+9Y/mN7eGrr2i4rm3kh/9y72PLP74Luqmq5qonls771dkNk5+9i6ifcYlDIPm++5Hrn1tw3nOLz+c57MEZqU+KeCYrLuwmlyWmKrQacYpBgVW+bia34/SDmSZvpfxAjJQ3gHakljzBdhLJKV/TJiYMwRXMLw0EF24y1YP0eG8C5gZm0JO/f5Jju9HhjUI6f/0oCSoeSx+x48Zq/qLYQxE1hjYPvB3Y/m/QWnIldWvpXYziJtcNChSlEoagUEIwBhOP2m9WMzKJHXczfxTMG3Rqad4qbb5pfh7cPUii3T+n59cv454kV5XKwGZ04Oolx6qDm6nMrD3Q4pBaWbiDuqXzgYd/S1VoCxmK+MkUt4W7adHrRkNI3jqzvvfh4oygfo0Ye+aseb9qOPtHF17pX7yGlsUhRhsxlfUDTnQzJ1B+aDvqkMK3g+CvaNr4i77vgcGUUUY96IM7LurjIp5Lc3FhlIW7tAQaMwlBZARuY9XCeG9nwvA4cmMC8+cy7kB3IT+r70orHCMX4Huo1ybD8G9ncv+UkBR0fmdT6SzYDMaN8nD9/OYcH3sXpH5TgjW0Gz+VSpM8ffMXf8TCOwtXvCEKeTCRVoXeVLkj1I9BQZAngtV1jdyDLiASDyq6YoaNYqVtk3Q9726i4lQ+nnoNRsUbaZm14Ha0mlDGg4Eet4WR1izcb5e2sVmXoBWtEPWVnNO31ig2NOgSVqeVd9+ER2ukaAWMYuph4c3TbFhsvSO8mYV7MFgo3M7kge1f+kFxrpOG00l9ELAKBrEMuwPRef6N5J7DAEzUjEXcbcEIKlJeNYU7b2ChwaTtzJMOHWOVq1htG5MH//28Tb9asuiFxUspskqZifo45Nj1aRL/qvZp8AA+MapR54YuqPA6xivSXfBJ27K47aUlnBvada0KHdqNONMwG84E/bsm/EzXqdZ8OeaEcW6oahtNpYvUKpX+1T4V2phCRprvpZz0vMXF6o+lCX8tjxZnkdH/j7wxrx3iBx5/9fVDPJ2PM9X2v0jZgy8YDWUuVu9GFRpMgVL5xPkR7br2CBS1htXTd/2ilu/U4HRSX+WTGZANIVR4DJ3vBFKgh6GLI3n3+c0Hzz73xUVncQzDmon6mjucOjqTp5wp5ceqFIjK+IHELU6KQbfpUCnQl0qedqNeWmlQW+hlRMV35aPs8g020f19QFTTaiz1IPnCVS2smZu2BSlEGdujdQL+TteLoQWQcqM0/e58J9B0M1Wn5MfiFOF0Uv/NUc7qO9AZl7ecTjzJ3WAbsHBieN5fHwv1EXj+/chZKzZ7pRuKUnXwJnTClB5RgJJSeVV4y/Qbq6S/phlVevy+8e0HvGJzZXD19BvfO6xl0ZhUjLniWo98nTe4fvqNlcEbqkOrG87fesfXHsbXt2B29Ny8LybHJyp8n/IE15dsD+ozOjcLSQc82yyxxR1aBdatR1prr/0EKk4lgFSWFpsKl1c5peYq8QrcvYJ5xFW372rj1+CaSnltRWhlcTEnDaeB+o+eu/zFhfPZnNUGcUNtMAHcvhGn2E3mVIdHiEkN8l1/t+w3P7j42XvO+9E/f+z8JaHa+nYWiLEQerXRnRfarEUsFqXGmk6kQHpm6pfw8JhSokLeqOaOQn2Vu8SrS69UUoeqYKxL8S1ESk+oraCzFSWnsBXdqkeReXY/Nulo9+arJkMoEJvVFL/i2r8h2Z8pNXu1K9hKJvTTRliJYr/yrV9wbPP02ykwgVyxWk6b1OuQe0dpuSH9qAC0771+c4QFmOndTmGAhD4+KQsaIaKYxH6H1GIq4eTiNFD/+Yammsp2tIHyz+z247brrNCusxfNf+X+ar6HKXtn8z327C+Y+pg7u2/xK3su+Nl3Lvjht/7ylYcuf+fZFR+9ZGlV4Gbcjwytn9J3+VS5+Gbo+hOnPkbUJd4m0VcEVQvklJJOweS2K6Tp1M/rA6/8kbMI2S0Y0FZ8o01MYuDDlIhmA8dEfT2hCcSCg15p5Md7Dk2Ma7pEEXBt8YibmUz7YsUlYLKHSm6BI4qojwlmXXUfaO5vv6UvXgSL+nnMFrYwYdu2sy66aM61LHSzIxit9IEITzhCUVD3hbnz+MHZY48y/mBVBqj/AHtr//K5TX/lCG7TfNswqE7/CLp9Qq0e/wCLJNADLcc9dbcyuXUKn9ATH2NiMwySK7CNBUaKm4KYkfropBuqlNfj4TpNH9UoqSJtahZszRvf02+cRn1dCcmwuk2FbNhySTs+kk8NyfrGOPkKzbfqmE59Cq7EBuA+ANBX1pMed5DP9r2fPc3JxWQqbAL+fxzP0xRqjznFKG42C0PGRbkV/VSlJn8J6kMKdTjrk4fGcmmV1B/8z6J+HizUxeTOtYtQJumOZ3GHU+xwh/sP72tUDzD+kC2138v3zvrpnde4mxIs3OLAI04d+g6lEGWB22x+CnYNt625vsUjdCOHIm2sctg+ZZ8S/XFIlLrVzlDiBKhPcQe4mUXaM22m4VhmM7l0Cjeheo4mvEtQnzyA6ACtM7Uw0m4XN5DrsHAvCmB09pfSMaZT3yVHv/n9g9/+wS//8dv7I8tb7HIn2voC5clHNGCq3YVlmovE72PLLjb215jYwsLDX/rOHxmGAOYvBoaqGzaUmoYlqU+Vyhjqo0ehIbMt6ufBGrbacYt0AwvTmUDayvYG2g7/J8sC73ez8cdA5M8Rg+uo63eRSziOAYYycdHX8djB/83EYSbuZI19Y79cUOXD6EKntPGCFecygULSMd4wRvKYhl9qc0V2gFVQ3BTETNTHiADcaYrNbYyPpjRXBg5VJs2r5ZWsviWvV0y7cTr1iQE18NQNRgQEC24ZxxcHdRSuOIIdLBRberFZNzMIO536LLw1S/JVW1J4LgWqhhsWK3mqwyCwpVLumKKykxfVZexVxT2+QSa2qzA5Ze0YGrUHVoD6xJFS3qJS1Mc+J90p5o4ktN06Tm5fPLZrzlae1HdJW5nYmxfJKJU99S2ZvY5JUHJ2V/GHq1IHmE0cdsi3wupJC7oe7wHiH3e1QA6Flv30zqtYOMmkzfyhahwzKf7m3lomrmehbTb/LrjFIYxM6U1KxU1BlKR+oW35K3U9B36Fp0M4ubFhUF3yjpnyl6C+goebGhIYm53P5g7uBD3q1VGT1kG7Fvagtp1JTDWJ3OnUt/sHFN3JaACYyqRBcwgn6lSBnpSpKEVNf+4ffsx8JveA0LPgozGuKs7AhsITwY3O0Ijbv4o4OsUEKUV9Izkjve5gL9X4LrSOyiz8alFfIlpLbYcfP5sj7238cZe6n80W+5AieKZBU6NjrLa7oiH23K8HFy5oBqFo9yXUvfK/fHGpM3i1coCxwHUv//wT3/rCOUwecoixpb7Gyz49xztvePp4FDcFcRTqC13M12lcCUS9kR58wQfuu6ScUidItcKvdrkwk/P5p1Ff5ZPnXATLSJLiFvVs518+DHmOjKWdMI0DpAiFovjUcs8X//4+TWQaLT026uOcUXgF2P2CIbwxjk3a9co7RlHZbBqjXMPGgsOCPcBmNcffPkIH1emJUPSEepi/O6fHQBglzEx9mmwdrC5K3uGM17/Z9GvZU98ZbGGhzTUN6/hepiX1Udd//sOlaMKa9FSvL0FvHBhk85sfv/fT7+yur6//9NbWpTBb+taAYcCiKxl/mIHJ+4k/+5TyZN284HIm3ArqDZDMJuzAUIh8UcVNQZSgPiwy9kiXdm7V2ZDEKGLUygZ+8yqyKquAuYIhk5pmBZ/jcAWVMZPePI366WzGLsXxPRxBCq3BbAm0IIm4VaFmjIDAX/USKiO9ipoqEufHQn38nuO+RQnmN5YXEiK9d9/7QiHb4RR3hndSxL+exx0eyihjnBQhDMUJ7DBb8Ou7vkZT0fBalqJ+DOM7ZNpZz190hWPjKW6rucacrdypjx0UHnj63ssK1Of7mKO2h4WbnUHTeYuaPhaOsUXbWaSHLY0vXriI7xGO3IuZ+R5nan8Ff9iJRsJDlYceYLaa9bZQG/O12Bp2srn9OIXwILleVHFTECWoT9xKpvg4nmkKDOAmA57lSzLfZshsD24CixmHXG5zR3a4A2uzmRmor4vt+x48bGscRIsF1zpcKOyhGA48kjaDR0/8fdpRPT2F+g5PomV8vAqPRv36hb22oKHMFFFfUZQqPG8waDiXAj03tn6FqxS1r/DK0OpiG8YX1bS9AqZTHwYUpECFeCMz+6xklB0u0/EGi/ooGNxCIrfHV6B+ar/X7m+ewgBIUsLp3/L5m5vS+3zk6WfqHuC9ne+1GXNmj23yl0w54J2EOfCwb+JJ4bLzFk4phFJxUxClqI8qR2wCbEZgpIAnAAs/tdyyj9VtIXGYdAZbWXgIRNoEz+gnmIw269THocUtngmMH9Z/woXCIcfW9d6TJgJBGgftfM5G8yxFeVy/AZ2qx0l9MhJSzvrNU3ymuPRF3x2nX/HtPKPGT5Cg3trmsXxj0nQeBQ3WKSVs/uFPDr2vwpOjRQP9OWIX9IlZuzOlcqW+W25laOrh2T9PoE094Dao/8RHQYd2ilGzBAU91VF/MwvtgIF3CDG5doWyuwT1c/vs//euWTUgcqBYYXDW3Fun9Xi8EIhiwkzUV3jmo58cdEomP7dOO9LshW2zpDVA6xmoT1w5MqmY3edYiHbukckdesIzVlGzJx6/y3FyERpa07FQH9SkN9/lJLONbNBaT0N/hsI2OZ8YuPXHpqZG9ZywsuntoWO4oY68rUXtgdVP2oxBdnmUon4MiUxtrhQ30obgVCmWz1am1PeG27VT2w4xDopNZneNQf3Hwqy+jTZ6TMOGw9DlCy9796k/57+AbBXZ/XawiXN73cB44r0dNR80lB2Zg06+nz15d4tTWIW6vr6zk9AWmeKmIGaifnocQ9lxJ8EYGL0oyOOav0sLA56J+vimy4lZonFuEHlPcaakMnXTa8/wJQXEM+OpUV7K8Wr/uqxC59oJJagv9GrUpzkNpMMo6yp5M5MH3EKhzRghAgIlm8MXZQJ7nXWmyay9tYoiZPX24C4bPY5puaPo/PhrhwptKUF9eAR4Zq1P0mrKI+LKY9b786lsqS8228LddCYNRcs7B+cVqD/5MGOBbQxtPlN/hTYw71rc90FB2Ds7tOa1Az7+ABC9IvcIyz3K8PMAG32cvbD7ahbqZvW7qoVOt6/P4V/H5Fa3b4TOB3a4wn3FTUHMRH1SbjOrtnxp2uBBSviXtINqkeXZGaifzqaAInZ5pz5hMNGDI33x7QZaYrSzO+WcHp1pdIWSZCvoQn069UFNylJwspLl45yf/Vf9aOrUkWVielMQE4bQpM7mspnUuxnuDt9iairF30t4BkVvDFQdGMTTg6boBlJ+km7/qoL1Uor6cbio2wMYOMhZ3Rom4lvfTI8fL1/qt8W/z6Q2HGYpDhbkDatWFKiv7rXXzG1wY1SMqafC3SjAZFgKbvfPOZc/X/XfD65dXDePH3Ac2j0H5kDqiQjcWxG6Nv3kRYcfnis03oaZ5REWxpfheAXtxS/RimBHcVMQM1IfkVIVju/kMY8NmXRZBecGjPEM1IcMc0JrwUw3KzNuaWgU3YuoUo+SYj2axTOKV274qqkKOr0e6r/0qjvyTc2WoD5+T6IqL8H8wTcH0kWt94xsrYlvcHqbcS47aatvwa26QgmRhMPfAtNmVDXaM5bBL8UCW4rbQgl4XtKa8LWHM1Gf49sG4Hu1vIZNnUXlS33czA/06ge0peSc2u054P1B0vj32dMHL2DzBl0NhpsP5Y0vzvwd//b1G5bPl1lTzCP1v3f/4pcOfsojLsvtdlZL14//8uObV5/jqd/pjmz56df+fHbNCnQqG72JB1U/uark643el/qI/U+N4+uXkUwD+AKc0Mjm+LfzP5aU+lFVxS2wVE5xR7ZPiYuU4nMiURXPpph89kiXFBq7egkGa6GX8GWuKPeVUtSfnjSFqhu1dhDn9dueesWoZzQDnT91QfAlH30BGjBhMoNwHQHtaO3WfzTrPFpyi+u4Cp2WeT/q6xjPKBXSBhY2NhnKl/qZTK5CvhFllZjEg5gNG5vql4Plqu51o49yH/vZ965x1xoveEJhQ1YBaqUR0Hx2VdZc+YdHr3I0dHt8t/DdLo9vI8g8fqCOReIePP6sxdUY6qwmvUxEM+OYqI+vj5U2G++OrV2fj1FBHI36mez48kuHWYhey2wq/Lk38NapDEF9HY+qiS14thBzGrbBl//pAB5lUkspPNMSCZQ4q4vPafps04Ub4Bmy40hIDQ7faiYPTpHBvlZqzJQNBHo0tGltfnoFi3kCCNE0qViozxwD9amHgf03mRpZrtSHx03nuEM2qZvB9jv6Lx59iPEHKnL7bHw3u++uyyuqk6wBXzCWV4713rfJPS8+voFJI3jGWe4f332uQxjx+AZn1aysqFrOIsWaCXVli9AUP+ohf/zbBofkxWu8UoeWKoJRd2CrOXcO9R7eEv/WpvjdLT3fv/+hl1KmIP6xHLqt3FK0UII3cF2WDD4vXAx242ehcKkFS55GfQ3f/MFLXrnFJZuKkjrmyjfQvM24hfWuULs3GDP/mq+xB5rt8DULjc0P7nuVCjdPdmViAg836o3BT/3GOZHrxjJva6uKKb+OivrrXXK3AyrN53fO3fIXV3arFLNdKSQdYaO18GgUeIfHl4ugqiq+matuk5bNHez1+j5NjUSz2z13daF80Etd/q0O/5riIk4aTiH1UQPMMt/1BWo6QjuY3HPfnZfxh718j1vZz/iBqpf3f8brS7B5bUXUZ2LU4e9zhUdAvbHP6zrwX9u0d7kw4WZPYxfzm4PD8uWLN6fwNR4lhlal4CrSg2notcQnsuqUv0SHx60Ukmek3HP62yMG0BqewKmVL0FRskqaDohoh2S1v/KgJaxPE3elTlEV8mgV0fecMq7dgh4a809FKacVDt9S0IApc4uu53Lq9PagRo7v2NFuLAadM6O/OZPPn1EmtJMocF8OhXyKrGxKUEsO9+FKHrVR0/lKVeiZCSgn34JsOvteoXy8kC7ZlpOFU0d97U3LKi5zeWoKIyzUWhXaZWOXTB5kwH7Q+7P70PD9xt9e4gzrrwPQ80v4CkEm7ERHuLzN7utHzbUhjn4hcaB4D5LSGxn0R5QYDa5xAmMq8V/tH7qiD0MB2QKxsprxWkws/N98VDNtetJSTvzTOGFKdKvm8i+GPgPN+VFh0Oan/pPmPSxO+eK00+7TkH96rXBTFTn9yaYcLTCg4s9YoJE/pbdHu1L8q1Z7qWczmkrHiykLFUCveSv8ihnyn6cEp476BcC6yLT3mOY5apOGmNweliP86XPUveitVx9h/NH63B52zz9v8EoX20LtdrHTEbrZKbR5ay5ZeeUV8y+6iAm32WTjNZdGCnXYQrfc98Dvoa5pL1i2YEHHaaB+NptO4d9cMFE/MOQWtDcxjbDgTfMblk0+uhx3bWEC7GHKIw6O+7jsjQPLLvvk5Sy8hkL5pwU+FJIQ/z29crm4YgsWTDgN1Kc1LQtqrC2UYLNbaIu3dCq41af716clfBE2q29zR4bpDVBTVV4LFqbhNFEf1Uz825Grtv49q9Rj2j5YSjhCfQ7/Sv08IRoVR9FiLVggnAbqc2R/Xhshe6t2YSvD15D0VjUNU+gOhS5jVAntyBb4re3S46lZerew5ucO9DMx7gjclEK3olZoka1qwUIJnB7qFwOF9MSrY9wrRPF4CsYztuLOF0Z3FUt32njSNrySjkBiz9PvodvkKL57CxaOhjOC+iSsUyptc/BxdJPd89PfsNqbzH/Kz5Rarmv7/IQm1tXsxDhQv6Rfz4KFmXBGUP8oyKSzGdJiMqnsWA6DJTU3Ne0ZWbDwwXAGUz+/zUHbJdlcYS/Ect1Y+DBwBlPfgoWTCYv6FsoUFvUtlCks6lsoU1jUt1CmsKhvoUxhUd9CmcKivoUyhUV9C2UKi/oWyhQW9S2UKSzqWyhTWNS3UKawqG+hTGFR30KZwqK+hTKFRX0LZQqL+hbKFBb1LZQpLOpbKFNY1LdQprCob6FMYVHfQpnCor6FMsX/A9/L/UM3Cu7TAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,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>