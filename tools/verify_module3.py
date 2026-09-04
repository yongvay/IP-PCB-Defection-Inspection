"""Module 3 self-check — is my part actually done?

Owner: Ng Zhi Xuan.

Run this before every checkpoint and before submission:

    python -m tools.verify_module3

It walks the seven tasks assigned to Module 3 in the workload plan and reports,
for each one, whether the deliverable exists, runs, and produces the output the
plan says it should. Nothing here is graded; the point is that "I think it
works" is replaced by a table that either passes or does not.

Checks that need the DeepPCB dataset are skipped with a message rather than
failed, so the script is meaningful on a laptop that has not downloaded 1,500
image pairs. Skipped is not the same as passed, and the summary says so.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "data" / "DeepPCB" / "PCBData"
OUTPUTS = REPO_ROOT / "outputs"

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str, str]] = []


def record(task: str, name: str, status: str, detail: str = "") -> None:
    results.append((task, name, status, detail))
    symbol = {PASS: "\u2713", FAIL: "\u2717", SKIP: "-"}[status]
    print(f"  {symbol} {name}" + (f" — {detail}" if detail else ""))


def check(task: str, name: str, dataset_required: bool = False):
    """Decorator turning a function that returns a detail string into a check."""
    def wrapper(function):
        if dataset_required and not DATASET.exists():
            record(task, name, SKIP, "DeepPCB not present")
            return function
        try:
            detail = function() or ""
            record(task, name, PASS, detail)
        except AssertionError as error:
            record(task, name, FAIL, str(error) or "assertion failed")
        except Exception as error:
            record(task, name, FAIL, f"{type(error).__name__}: {error}")
        return function
    return wrapper


# ---------------------------------------------------------------------------
print("\nEnvironment")
# ---------------------------------------------------------------------------
@check("env", "Every dependency imports")
def _dependencies():
    missing = []
    for module in ("cv2", "numpy", "pandas", "skimage", "reportlab", "streamlit"):
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    assert not missing, f"missing: {', '.join(missing)}. Run pip install -r requirements.txt"
    return "cv2, numpy, pandas, skimage, reportlab, streamlit"


@check("env", "Every Module 3 file is present")
def _files_present():
    expected = [
        "src/module3/descriptors.py", "src/module3/connectivity.py",
        "src/module3/classify.py", "src/module3/evaluate.py",
        "src/module3/pdf_report.py", "dashboard.py",
        "tests/test_module3.py", "experiments/benchmark_module3.py",
    ]
    missing = [path for path in expected if not (REPO_ROOT / path).exists()]
    assert not missing, f"missing: {', '.join(missing)}"
    return f"{len(expected)} files"


@check("env", "The dashboard imports without crashing")
def _dashboard_imports():
    # Imported as source rather than executed, because running it starts a
    # Streamlit server. A syntax error or a bad import is what this catches,
    # and a bad import is exactly the failure that was in the repository.
    source = (REPO_ROOT / "dashboard.py").read_text()
    compile(source, "dashboard.py", "exec")
    for line in source.splitlines():
        if line.startswith("from src.") or line.startswith("import src."):
            module = line.split()[1]
            importlib.import_module(module)
    return "all src imports resolve"


# ---------------------------------------------------------------------------
print("\nRegression tests")
# ---------------------------------------------------------------------------
@check("all", "tests/test_module3.py passes")
def _module3_tests():
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_module3.py", "-q"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout.strip().splitlines()[-1]
    return completed.stdout.strip().splitlines()[-1]


@check("all", "The rest of the suite still passes")
def _whole_suite():
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert completed.returncode == 0, "Module 3 changes broke another member's tests"
    return completed.stdout.strip().splitlines()[-1]


# ---------------------------------------------------------------------------
print("\nTask deliverables")
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO_ROOT))
from tests.test_module3 import board_with, context_for, localise, reference_board  # noqa: E402

from src.contracts import DEFECT_CLASSES, Defect  # noqa: E402
from src.module3 import classify, descriptors  # noqa: E402
from src.module3.evaluate import Evaluation  # noqa: E402
from src.module3.pdf_report import generate_pdf_report  # noqa: E402

ALL_CLASSES = list(DEFECT_CLASSES)


@check("3.1", "Every descriptor the plan names is computed")
def _descriptors():
    template, test = reference_board(), board_with("pin_hole")
    blob = localise(template, test)[0]
    measured = classify.describe(blob)
    promised = ("area_px", "perimeter_px", "aspect_ratio", "solidity",
                "extent", "eccentricity", "hu_moments")
    missing = [name for name in promised if name not in measured]
    assert not missing, f"missing: {', '.join(missing)}"
    assert len(measured["hu_moments"]) == 7, "Hu moments should be seven values"
    return ", ".join(promised)


@check("3.2", "Stage one recovers polarity from the difference sign")
def _stage_one():
    template = reference_board()
    expected = {"open_circuit": "removed", "mouse_bite": "removed",
                "pin_hole": "removed", "short": "added", "spur": "added",
                "spurious_copper": "added"}
    for defect_class, polarity in expected.items():
        blob = localise(template, board_with(defect_class))[0]
        assert blob.polarity == polarity, f"{defect_class} read as {blob.polarity}"
    return "6 of 6 correct"


@check("3.2", "Stage two assigns all six classes")
def _stage_two():
    template = reference_board()
    correct = {"connectivity": 0, "descriptor": 0}
    for defect_class in ALL_CLASSES:
        test = board_with(defect_class)
        blob = localise(template, test)[0]
        context = context_for(template, test)
        for method, ctx in (("connectivity", context), ("descriptor", None)):
            predicted, _, _ = classify.classify(blob, ctx, method)
            correct[method] += predicted == defect_class

    assert correct["connectivity"] == 6, (
        f"connectivity got {correct['connectivity']} of 6 on the fixture"
    )
    return (f"connectivity {correct['connectivity']}/6, "
            f"descriptor baseline {correct['descriptor']}/6")


@check("3.2", "Confidence is graded, not a constant")
def _confidence():
    template = reference_board()
    values = set()
    for defect_class in ALL_CLASSES:
        test = board_with(defect_class)
        blob = localise(template, test)[0]
        _, confidence, _ = classify.classify(blob, context_for(template, test))
        assert 0.5 <= confidence <= 1.0, f"confidence {confidence} out of range"
        values.add(confidence)
    assert len(values) > 1, "every defect scored the same confidence — still a placeholder"
    return f"{len(values)} distinct values across 6 defects"


@check("3.3", "Area converts with the calibration factor squared")
def _measurement():
    blob = localise(reference_board(), board_with("pin_hole"))[0]
    area_mm2 = classify.measure(blob, 1 / 48)
    assert abs(area_mm2 - blob.area_px / (48 ** 2)) < 1e-9, "linear factor applied once"

    features = classify.describe(blob)
    length_mm, width_mm = classify.measure_dimensions(features, 1 / 48)
    assert length_mm >= width_mm > 0, "dimensions are not ordered or are zero"
    return f"{blob.area_px} px -> {area_mm2:.4f} mm2, {length_mm:.3f} x {width_mm:.3f} mm"


@check("3.4", "The verdict is severity-weighted and explains itself")
def _verdict():
    def make(defect_class):
        return Defect(id=0, bbox=(0, 0, 5, 5), defect_class=defect_class,
                      area_mm2=0.01, confidence=0.9,
                      severity=classify.SEVERITY[defect_class])

    clean, _ = classify.decide_verdict([])
    assert clean == "PASS", "a clean board should pass"

    critical, detail = classify.decide_verdict([make("short")], {"max_defects": 99})
    assert critical == "FAIL", "a short should fail regardless of tolerance"
    assert detail["reason"], "the verdict gives no reason"

    cosmetic, _ = classify.decide_verdict([make("spur")], {"max_defects": 2})
    assert cosmetic == "PASS", "a tolerated cosmetic defect should pass"
    return "critical override, count tolerance and severity limit all fire"


@check("3.5", "The dashboard runs the pipeline rather than recomputing")
def _dashboard_sources_results():
    source = (REPO_ROOT / "dashboard.py").read_text()
    assert "inspect_pair" in source, "the dashboard does not call the pipeline"
    for banned in ("cv2.threshold", "cv2.morphologyEx", "connectedComponents"):
        assert banned not in source, (
            f"the dashboard calls {banned} itself — what is on screen is then "
            "not what the harness scores"
        )
    return "no image processing of its own"


@check("3.6", "Summary panels cover count, class, area and verdict")
def _panels():
    source = (REPO_ROOT / "dashboard.py").read_text()
    required = {
        "verdict banner": "verdict_reason",
        "class distribution": "value_counts",
        "area distribution": "Area (mm",
        "defect table": "dataframe",
        "severity": "Severity",
    }
    missing = [name for name, token in required.items() if token not in source]
    assert not missing, f"missing panels: {', '.join(missing)}"
    return ", ".join(required)


@check("3.7", "The harness scores localisation and classification separately")
def _harness():
    evaluation = Evaluation(iou_threshold=0.5)
    evaluation.add_board(
        "check",
        predictions=[((0, 0, 10, 10), "short")],
        truths=[((0, 0, 10, 10), "spur")],
        runtime_s=0.15,
    )
    assert evaluation.localisation.f1 == 1.0, "a found defect should localise"
    assert evaluation.classification.f1 == 0.0, "a misnamed defect should not classify"

    summary = evaluation.summary()
    for key in ("loc_f1", "cls_f1", "macro_f1", "class_accuracy",
                "mean_runtime_s", "within_3s_pct"):
        assert key in summary, f"summary is missing {key}"
    assert len(evaluation.per_class()) == 6, "per-class table should cover six classes"
    return "localisation, classification, macro F1, class accuracy, runtime"


@check("3.7", "Metrics export to CSV for Xing Szen's figures")
def _csv_export():
    evaluation = Evaluation()
    evaluation.add_board("check", [((0, 0, 10, 10), "spur")],
                         [((0, 0, 10, 10), "spur")], runtime_s=0.2)
    with tempfile.TemporaryDirectory() as directory:
        written = evaluation.write_csv(Path(directory), prefix="check")
        assert len(written) == 4, f"expected 4 tables, wrote {len(written)}"
        for path in written:
            assert path.stat().st_size > 0, f"{path.name} is empty"
    return "per_class, confusion, per_board, summary"


@check("extra", "The PDF report generates and is non-trivial")
def _pdf():
    template, test = reference_board(), board_with("short")
    blob = localise(template, test)[0]
    defect_class, confidence, decided_by = classify.classify(
        blob, context_for(template, test)
    )
    features = classify.describe(blob)
    length_mm, width_mm = classify.measure_dimensions(features, 1 / 48)

    from src.contracts import InspectionReport
    report = InspectionReport(
        defects=[Defect(id=0, bbox=blob.bbox, defect_class=defect_class,
                        area_mm2=classify.measure(blob, 1 / 48),
                        confidence=confidence, polarity=blob.polarity,
                        width_mm=length_mm, height_mm=width_mm,
                        severity=classify.SEVERITY[defect_class],
                        decided_by=decided_by)],
        verdict="FAIL", runtime_s=0.2, mm_per_px=1 / 48,
        verdict_reason="1 critical defect(s) present: short",
        classifier="connectivity",
    )
    data = generate_pdf_report(report, ssim_score=0.98, board_name="self_check")
    assert data.startswith(b"%PDF"), "output is not a PDF"
    assert len(data) > 2000, f"PDF is suspiciously small at {len(data)} bytes"
    return f"{len(data)} bytes"


# ---------------------------------------------------------------------------
print("\nHousekeeping")
# ---------------------------------------------------------------------------
@check("hygiene", "No AI drafting artefacts left in the source")
def _artefacts():
    # Built at run time rather than written as a literal, so that this file
    # does not match its own check.
    marker = "[" + "cite:"
    offenders = []
    for path in REPO_ROOT.rglob("*.py"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if marker in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"'{marker}' found in: {', '.join(offenders)}"
    return "clean"


@check("hygiene", "No committed bytecode")
def _bytecode():
    stray = [str(path.relative_to(REPO_ROOT))
             for path in REPO_ROOT.rglob("*.pyc")
             if ".git" not in path.parts and "__pycache__" not in path.parts]
    assert not stray, f"delete: {', '.join(stray)}"
    return "clean"


def _prose_of(path: Path) -> str:
    """Extract comments and docstrings, discarding code identifiers.

    Spell-checking raw source is useless: cv2.COLOR_BGR2GRAY, colors.HexColor
    and the reportlab alignment constant "CENTER" are API names, not
    misspellings, and flagging them trains the reader to ignore the check. Only
    prose the team actually wrote is examined.
    """
    import ast
    import io
    import re
    import tokenize

    source = path.read_text(encoding="utf-8")
    prose = []

    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            prose.append(token.string)

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            if docstring:
                prose.append(docstring)

    text = "\n".join(prose)
    # Drop screaming-snake-case constants and dotted API calls before folding
    # to lower case, so that COLOR_BGR2GRAY cannot masquerade as "color".
    text = re.sub(r"\b[A-Z][A-Z0-9_]{2,}\b", " ", text)
    text = re.sub(r"\b\w+\.\w+\b", " ", text)
    return text.lower()


@check("hygiene", "No US spellings in the Module 3 prose")
def _uk_english():
    # The rubric marks UK English on the report, not the code, but mixed
    # spellings in comments read badly when the marker opens the source.
    import re

    us_spellings = ("color", "colors", "gray", "normalize", "normalized",
                    "analyze", "analyzed", "behavior", "center", "centered",
                    "optimize", "optimized", "recognize", "labeled", "modeling")
    offenders = []
    for name in ("descriptors.py", "connectivity.py", "classify.py",
                 "evaluate.py", "pdf_report.py"):
        prose = _prose_of(REPO_ROOT / "src" / "module3" / name)
        found = sorted({word for word in us_spellings
                        if re.search(rf"\b{word}\b", prose)})
        if found:
            offenders.append(f"{name}: {', '.join(found)}")
    assert not offenders, "; ".join(offenders)
    return "UK English throughout"


# ---------------------------------------------------------------------------
print("\nAgainst the real dataset")
# ---------------------------------------------------------------------------
@check("3.7", "Benchmark outputs exist for Chapter 4", dataset_required=True)
def _benchmark_outputs():
    expected = ["module3_classifier_comparison.csv",
                "module3_connectivity_per_class.csv",
                "module3_connectivity_confusion.csv"]
    missing = [name for name in expected if not (OUTPUTS / name).exists()]
    assert not missing, (
        "not yet generated — run: python -m experiments.benchmark_module3"
    )
    return ", ".join(expected)


# ---------------------------------------------------------------------------
def summarise() -> int:
    passed = sum(1 for *_, status, _ in results if status == PASS)
    failed = [row for row in results if row[2] == FAIL]
    skipped = [row for row in results if row[2] == SKIP]

    print("\n" + "=" * 72)
    print(f"{passed} passed, {len(failed)} failed, {len(skipped)} skipped")

    if failed:
        print("\nFailed:")
        for task, name, _, detail in failed:
            print(f"  [{task}] {name}: {detail}")

    if skipped:
        print("\nSkipped — these are NOT passes:")
        for task, name, _, detail in skipped:
            print(f"  [{task}] {name}: {detail}")

    if not failed and not skipped:
        print("\nEvery check passed, including against the real dataset.")
    elif not failed:
        print("\nEvery runnable check passed. Download DeepPCB and run "
              "experiments/benchmark_module3.py to close the rest — the "
              "synthetic fixture proves the rules are correct, but it is not "
              "evidence for Chapter 4.")

    print("=" * 72 + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(summarise())
