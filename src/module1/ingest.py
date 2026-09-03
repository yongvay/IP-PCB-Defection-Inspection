"""Data ingestion and dataset indexing — Module 1, tasks 1.1 and 1.3.

Owner: Chan Xing Szen.

Everything that turns a directory of files on disk into pairs the pipeline can
consume lives here. Keeping it separate from preprocess.py matters because the
two datasets are indexed completely differently — DeepPCB ships explicit
split files, HRIPCB has to be paired by filename convention — while the
preprocessing applied afterwards is identical. Mixing the two concerns in one
file is how a loader change silently breaks a filter benchmark.

Two datasets, two very different shapes:

  DeepPCB   640 x 640, greyscale, already binarised by the dataset authors,
            template and test pre-aligned, annotations as plain text.
  HRIPCB    colour photographs up to 3034 x 1586, ordinary greyscale content,
            ten shared reference boards, annotations as PASCAL VOC XML, and a
            rotated copy of every image with the applied angle recorded.

Author: Chan Xing Szen (Member A, Module 1)
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPPCB_ROOT = REPO_ROOT / "data" / "DeepPCB" / "PCBData"
HRIPCB_ROOT = REPO_ROOT / "data" / "HRIPCB" / "PCB_DATASET"

# HRIPCB names its classes in the folder names; the pipeline names them in
# src/contracts.py. Translating here means no other file has to know that the
# two datasets disagree about what a pin hole is called.
HRIPCB_CLASS_TO_CONTRACT = {
    "missing_hole": "pin_hole",
    "mouse_bite": "mouse_bite",
    "open_circuit": "open_circuit",
    "short": "short",
    "spur": "spur",
    "spurious_copper": "spurious_copper",
}


@dataclass(frozen=True)
class Pair:
    """One inspectable unit: a reference board, a board under test, its answer key.

    ``angle_deg`` is populated only for the HRIPCB rotation set, where the
    dataset records the angle each board was turned through. That figure is the
    ground truth the rectification experiment (task 1.8) is scored against.
    """

    name: str
    dataset: str                      # "deeppcb" or "hripcb"
    template_path: Path
    test_path: Path
    annotation_path: Path | None = None
    angle_deg: float | None = None


# ---------------------------------------------------------------------------
# Loading (task 1.3)
# ---------------------------------------------------------------------------
def load_grey(path: str | Path) -> np.ndarray:
    """Read any image as a single-channel 8-bit array.

    cv2.imread returns None rather than raising when a path is wrong, and a
    None propagating into a filter produces an error hundreds of lines away
    from its cause. Failing here keeps the message next to the mistake.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_colour(path: str | Path) -> np.ndarray:
    """Read an image in colour. Used only where hue carries information."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_pair(template_path: str | Path,
              test_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one template-test pair as greyscale arrays.

    The two images are not forced to a common size here. Whether a size
    mismatch is an error or something rectification should fix depends on the
    dataset, so that decision belongs to preprocess_pair, not to the loader.
    """
    return load_grey(template_path), load_grey(test_path)


# ---------------------------------------------------------------------------
# DeepPCB indexing
# ---------------------------------------------------------------------------
def index_deeppcb(root: str | Path = DEEPPCB_ROOT,
                  split: str | None = None,
                  limit: int | None = None) -> list[Pair]:
    """Index DeepPCB pairs, optionally restricted to a published split.

    ``split`` is "trainval", "test", or None for everything. Using the split
    files rather than a directory glob is a methodological point, not a
    convenience: parameters tuned and reported on the same images overstate
    performance, and DeepPCB ships the division its own authors used. Chapter 3
    should state that tuning used trainval and every reported figure used test.

    Each line of a split file names the image stem, and the two images of the
    pair are that stem with _temp and _test appended.
    """
    root = Path(root)
    if split is None:
        pairs = _index_deeppcb_all(root)
    else:
        pairs = _index_deeppcb_split(root, split)

    if not pairs:
        raise FileNotFoundError(
            f"No DeepPCB pairs found under {root}. The dataset is not committed; "
            f"see the README for the expected layout."
        )
    return pairs[:limit] if limit else pairs


def _index_deeppcb_split(root: Path, split: str) -> list[Pair]:
    listing = root / f"{split}.txt"
    if not listing.exists():
        raise FileNotFoundError(f"No such DeepPCB split file: {listing}")

    pairs = []
    for line in listing.read_text().splitlines():
        if not line.strip():
            continue
        image_field, annotation_field = line.split()
        stem = Path(image_field).stem                     # e.g. 20085000
        directory = root / Path(image_field).parent
        pairs.append(Pair(
            name=stem,
            dataset="deeppcb",
            template_path=directory / f"{stem}_temp.jpg",
            test_path=directory / f"{stem}_test.jpg",
            annotation_path=root / annotation_field,
        ))
    return pairs


def _index_deeppcb_all(root: Path) -> list[Pair]:
    pairs = []
    for template in sorted(root.glob("group*/*/*_temp.jpg")):
        stem = template.name.replace("_temp.jpg", "")
        test = template.with_name(f"{stem}_test.jpg")
        if not test.exists():
            continue
        annotation = template.parent.parent / f"{template.parent.name}_not" / f"{stem}.txt"
        pairs.append(Pair(
            name=stem,
            dataset="deeppcb",
            template_path=template,
            test_path=test,
            annotation_path=annotation if annotation.exists() else None,
        ))
    return pairs


# ---------------------------------------------------------------------------
# HRIPCB indexing
# ---------------------------------------------------------------------------
def index_hripcb(root: str | Path = HRIPCB_ROOT,
                 rotated: bool = False,
                 limit: int | None = None) -> list[Pair]:
    """Index HRIPCB pairs, drawing the reference board from PCB_USED.

    HRIPCB does not ship one template per image. It ships ten reference boards
    and names every defect image after the board it was made from, so
    ``01_missing_hole_01.jpg`` is board ``01`` with missing holes added. The
    prefix before the first underscore is therefore the whole pairing rule.

    This matters more than it looks. The team's working notes recorded HRIPCB
    as having no template images, which would have ruled out the golden-template
    method on the entire secondary dataset. PCB_USED holds all ten, so the
    method applies unchanged and only the resolution and colour differ.

    With ``rotated=True`` the test image is taken from the rotation set and the
    applied angle is attached to the Pair, which is what makes the rectification
    experiment measurable rather than merely illustrative.
    """
    root = Path(root)
    templates = {path.stem.lstrip("0") or "0": path
                 for path in sorted((root / "PCB_USED").glob("*.JPG"))}
    if not templates:
        raise FileNotFoundError(f"No reference boards found in {root / 'PCB_USED'}")

    pairs = []
    for class_dir in sorted((root / "images").iterdir()):
        if not class_dir.is_dir():
            continue
        angles = _read_angles(root / "rotation" / f"{class_dir.name}_angles.txt") if rotated else {}

        for image_path in sorted(class_dir.glob("*.jpg")):
            board = image_path.stem.split("_")[0].lstrip("0") or "0"
            template = templates.get(board)
            if template is None:
                continue

            if rotated:
                test_path = root / "rotation" / f"{class_dir.name}_rotation" / image_path.name
                if not test_path.exists():
                    continue
            else:
                test_path = image_path

            annotation = root / "Annotations" / class_dir.name / f"{image_path.stem}.xml"
            pairs.append(Pair(
                name=image_path.stem,
                dataset="hripcb",
                template_path=template,
                test_path=test_path,
                annotation_path=annotation if annotation.exists() else None,
                angle_deg=angles.get(image_path.stem),
            ))

    if not pairs:
        raise FileNotFoundError(f"No HRIPCB pairs found under {root}")
    return pairs[:limit] if limit else pairs


def _read_angles(path: Path) -> dict[str, float]:
    """Read one '<image stem><tab><degrees>' listing from the rotation set."""
    if not path.exists():
        return {}
    angles = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            angles[parts[0]] = float(parts[1])
    return angles


def parse_voc_annotation(path: str | Path) -> list[dict]:
    """Parse one PASCAL VOC XML file into the same shape as the DeepPCB parser.

    Returning the identical dictionary shape as ground_truth.parse_annotation_file
    is the point of this function. Every downstream consumer — the evaluation
    harness, the benchmarks, the figures — then works on either dataset without
    a single branch on which one it was given.
    """
    root = ElementTree.parse(str(path)).getroot()
    stem = Path(path).stem

    boxes = []
    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip().lower()
        label = HRIPCB_CLASS_TO_CONTRACT.get(raw_name)
        if label is None:
            raise ValueError(f"{path}: unknown HRIPCB class '{raw_name}'")

        box = obj.find("bndbox")
        x1 = int(float(box.findtext("xmin")))
        y1 = int(float(box.findtext("ymin")))
        x2 = int(float(box.findtext("xmax")))
        y2 = int(float(box.findtext("ymax")))

        boxes.append({
            "file": stem,
            "bbox": (x1, y1, x2 - x1, y2 - y1),
            "label": label,
            "polarity": _POLARITY[label],
        })
    return boxes


# Imported lazily from ground_truth to keep one definition of the mapping.
from src.module1.ground_truth import POLARITY as _POLARITY  # noqa: E402


def main() -> None:
    """Report what is present on disk. Run: python -m src.module1.ingest"""
    for name, index in (("DeepPCB trainval", lambda: index_deeppcb(split="trainval")),
                        ("DeepPCB test", lambda: index_deeppcb(split="test")),
                        ("HRIPCB upright", index_hripcb),
                        ("HRIPCB rotated", lambda: index_hripcb(rotated=True))):
        try:
            pairs = index()
            extra = ""
            if pairs[0].dataset == "hripcb" and pairs[0].angle_deg is not None:
                known = sum(1 for p in pairs if p.angle_deg is not None)
                extra = f", {known} with a recorded angle"
            print(f"{name:<18} {len(pairs):>5} pairs{extra}")
        except FileNotFoundError as error:
            print(f"{name:<18} unavailable — {error}")


if __name__ == "__main__":
    main()
