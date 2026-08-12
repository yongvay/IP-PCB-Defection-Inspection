"""Integration spine — Task 2.5.

Owner: Ng Yong Vay.

The single entry point that runs Module 1 -> Module 2 -> Module 3 for one
template-test pair. Nothing here contains image-processing logic of its own:
its whole purpose is to move data across the three module boundaries in the
shapes fixed by src/contracts.py. That is deliberate, and it is what allows
any one module to be rewritten without the other two noticing.

Run it directly to inspect a single pair:

    python -m src.pipeline data/DeepPCB/PCBData/group00041/00041/00041000_temp.jpg \\
                           data/DeepPCB/PCBData/group00041/00041/00041000_test.jpg
"""

import argparse
import time
from typing import Any

from src.contracts import InspectionReport, LocalisationResult
from src.module1 import morphology, preprocess
from src.module2 import blobs as blob_extraction
from src.module2 import difference, registration
from src.module3 import classify


def inspect_pair(template_path: str,
                 test_path: str,
                 params: dict[str, Any] | None = None) -> InspectionReport:
    """Run the complete inspection for one board."""
    params = params or {}
    started_at = time.perf_counter()

    # --- Module 1: preprocessing and calibration -------------------------
    prepared = preprocess.preprocess_pair(template_path, test_path, params)

    # --- Module 2: registration ------------------------------------------
    # DeepPCB pairs arrive pre-aligned, so registration should be close to the
    # identity transform. It is still run rather than skipped, because the
    # residual is the system's own evidence that the pair is safe to compare,
    # and because HRIPCB boards are not pre-aligned at all.
    alignment = registration.register(
        prepared.test_bin,
        prepared.template_bin,
        method=params.get("registration_method", "orb"),
    )

    # --- Module 2: differencing and blob extraction ----------------------
    removed, added = difference.signed_difference(prepared.template_bin, alignment.aligned)

    # Morphological cleanup is Module 1's function, called from here so that
    # the tuning stays with its owner while the ordering stays with the spine.
    removed = morphology.clean_difference(removed, params)
    added = morphology.clean_difference(added, params)

    candidates = blob_extraction.extract_blobs(
        removed, added,
        min_area=params.get("min_blob_area", blob_extraction.MIN_BLOB_AREA_PX),
    )

    localisation = LocalisationResult(
        blobs=candidates,
        align_residual=alignment.residual,
        registration_ok=alignment.ok,
        method=alignment.method,
    )

    # --- Module 3: classification, measurement and verdict ---------------
    return classify.build_report(localisation, prepared.mm_per_px, started_at, params)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one PCB template-test pair.")
    parser.add_argument("template", help="path to the defect-free template image")
    parser.add_argument("test", help="path to the image under inspection")
    args = parser.parse_args()

    report = inspect_pair(args.template, args.test)

    print(f"Verdict:   {report.verdict}")
    print(f"Defects:   {len(report.defects)}")
    print(f"Residual:  {report.align_residual:.3f} px")
    print(f"Runtime:   {report.runtime_s:.3f} s")
    for defect in report.defects[:20]:
        print(f"  #{defect.id:<3} {defect.defect_class:<16} "
              f"{defect.bbox}  {defect.area_mm2:.4f} mm^2")


if __name__ == "__main__":
    main()
