"""Ground truth parser for DeepPCB annotation files (Module 1, task 1.2).

Reads the hand-marked answer key that accompanies each test image and
converts it into a structure the rest of the system can compute with.

Author: Chan Xing Szen (Member A, Module 1)
"""

from collections import Counter
from pathlib import Path

PCB_DATA = Path(__file__).resolve().parents[2] / "data" / "DeepPCB" / "PCBData"

# From the DeepPCB README: 1-open, 2-short, 3-mousebite,
# 4-spur, 5-copper, 6-pin-hole. Verified against the repository.
TYPE_TO_CLASS = {
    1: "open_circuit",
    2: "short",
    3: "mouse_bite",
    4: "spur",
    5: "spurious_copper",
    6: "pin_hole",
}

# Does the defect add copper to the board, or remove it? Read by the
# stage-one classifier in Module 3, so it is defined only here.
POLARITY = {
    "open_circuit": "removed",
    "mouse_bite": "removed",
    "pin_hole": "removed",
    "short": "added",
    "spur": "added",
    "spurious_copper": "added",
}


def parse_annotation_file(path):
    """Read one annotation file and return a list of defect boxes.

    Each line is 'x1 y1 x2 y2 type'. Corner coordinates are converted to
    (x, y, width, height) so the output matches the Blob and Defect
    contracts used by Modules 2 and 3.
    """
    boxes = []

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 values, got {len(parts)}")

        x1, y1, x2, y2, type_code = (int(part) for part in parts)

        if type_code not in TYPE_TO_CLASS:
            raise ValueError(f"{path}:{line_number}: unknown type code {type_code}")

        label = TYPE_TO_CLASS[type_code]
        boxes.append({
            "file": path.stem,
            "bbox": (x1, y1, x2 - x1, y2 - y1),
            "label": label,
            "polarity": POLARITY[label],
        })

    return boxes


def parse_all(pcb_data=PCB_DATA):
    """Parse every annotation file under a PCBData directory."""
    files = sorted(Path(pcb_data).glob("*/*_not/*.txt"))
    if not files:
        raise FileNotFoundError(f"No annotation files found under {pcb_data}")

    boxes = []
    for path in files:
        boxes.extend(parse_annotation_file(path))
    return files, boxes


def main():
    files, boxes = parse_all()
    print(f"Parsed {len(boxes)} defects across {len(files)} files\n")

    class_counts = Counter(box["label"] for box in boxes)
    print("Defects per class:")
    for label in sorted(class_counts):
        print(f"  {label:<18} {class_counts[label]}")

    removed = sum(1 for box in boxes if box["polarity"] == "removed")
    print(f"\nCopper removed: {removed}")
    print(f"Copper added:   {len(boxes) - removed}")


if __name__ == "__main__":
    main()