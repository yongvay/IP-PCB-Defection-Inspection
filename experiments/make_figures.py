"""Chapter 4 result figures — Module 1, task 1.11.

Owner: Chan Xing Szen.

Every chart in Chapter 4 is generated from a metrics CSV by this script and
never drawn by hand, so a figure cannot drift away from the numbers it claims to
show. Rerunning a benchmark and rerunning this script is the whole update path.

Design rules applied, and why they are rules
--------------------------------------------
* One quantity per axis, and never two y-scales on one chart. A second scale
  lets the author place the crossing point wherever the story needs it.
* Categorical colours are assigned in a fixed order from a palette checked for
  colour-vision deficiency, so a reader with deuteranopia or protanopia can
  still separate adjacent series. Series are also direct-labelled, so identity
  never depends on colour alone — which matters again when the report is
  printed in greyscale.
* Value labels sit in ink, not in the series colour. Colour carries identity;
  text carries the number.
* Grid lines are recessive and behind the marks. The data should be the
  darkest thing on the page.

Run:  python -m experiments.make_figures
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")           # no display in the environment this runs in

import matplotlib.pyplot as plt
import pandas as pd

from experiments.common import FIGURE_DIR, OUTPUT_DIR

# Validated categorical palette: worst adjacent CVD separation dE 9.1, worst
# adjacent normal-vision separation dE 22.9, both above the required floors.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dcdbd6"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9,
    "text.color": INK, "axes.labelcolor": INK_SOFT, "axes.edgecolor": GRID,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def _style(axis, title: str, ylabel: str, ymax: float | None = None) -> None:
    axis.set_title(title, color=INK, pad=8, loc="left", fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    if ymax:
        axis.set_ylim(0, ymax)


def _save(figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    figure.savefig(path)
    plt.close(figure)
    print(f"  {path.relative_to(FIGURE_DIR.parents[1])}")


def _read(name: str) -> pd.DataFrame | None:
    path = OUTPUT_DIR / name
    if not path.exists():
        print(f"  skipped {name} — run its benchmark first")
        return None
    return pd.read_csv(path)


def figure_denoise() -> None:
    """SSIM by filter and noise condition: the task 1.4 comparison."""
    frame = _read("denoise_benchmark.csv")
    if frame is None:
        return

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    figure.subplots_adjust(wspace=0.26)
    for axis, (noise, label) in zip(axes, [("gaussian", "Gaussian noise (sigma)"),
                                           ("salt_pepper", "Salt-and-pepper noise (rate)")]):
        subset = frame[frame["noise"] == noise]
        levels = sorted(subset["level"].unique())
        filters = ["none", "gaussian", "median", "bilateral"]
        width = 0.8 / len(filters)

        for index, name in enumerate(filters):
            values = [subset[(subset["level"] == level) & (subset["filter"] == name)]["ssim"].iloc[0]
                      for level in levels]
            positions = [i + index * width - 0.4 + width / 2 for i in range(len(levels))]
            axis.bar(positions, values, width * 0.88, label=name,
                     color=SERIES[index], edgecolor=SURFACE, linewidth=1.2)
            for x, value in zip(positions, values):
                axis.text(x, value + 0.015, f"{value:.2f}", ha="center",
                          fontsize=6.2, color=INK_SOFT, rotation=90)

        axis.set_xticks(range(len(levels)))
        axis.set_xticklabels([f"{level:g}" for level in levels])
        axis.set_xlabel(label)
        _style(axis, "", "SSIM against the clean image", 1.18)

    axes[0].set_title("Structural similarity after filtering, by noise type",
                      color=INK, pad=10, loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=4, loc="upper left",
                   bbox_to_anchor=(0, -0.22), fontsize=8)
    _save(figure, "fig_denoise_ssim.png")


def figure_pipeline(dataset: str) -> None:
    """Detection F1 by preprocessing choice: the SMART Objective 2 evidence."""
    frame = _read(f"pipeline_benchmark_{dataset}.csv")
    if frame is None:
        return
    if "split" in frame.columns and dataset == "deeppcb":
        # Report on the held-out split only; trainval rows exist for tuning.
        frame = frame[frame["split"].fillna("test") == "test"]

    stages = ["denoise", "enhance", "binarise", "morph_open_kernel"]
    titles = {"denoise": "Noise removal (1.4)", "enhance": "Contrast enhancement (1.5)",
              "binarise": "Binarisation (1.6)", "morph_open_kernel": "Opening kernel (1.10)"}
    present = [stage for stage in stages if (frame["stage"] == stage).any()]
    if not present:
        return

    figure, axes = plt.subplots(1, len(present), figsize=(3.6 * len(present), 3.5))
    axes = [axes] if len(present) == 1 else list(axes)
    figure.subplots_adjust(wspace=0.75)

    # The kernel size is ordinal, so its panel keeps numeric order rather than
    # being sorted by score: a sweep is read as a curve with an optimum, and
    # re-sorting it by result destroys the shape that makes the optimum visible.
    ordinal = {"morph_open_kernel"}
    # Chosen on the tuning split. Highlighting the winner on the reported split
    # instead would be exactly the peeking the split exists to prevent.
    selected = {"morph_open_kernel": {"deeppcb": "5", "hripcb": "5"}[dataset]}

    for axis, stage in zip(axes, present):
        subset = frame[frame["stage"] == stage].copy()
        subset["setting"] = subset["setting"].astype(str)
        if stage in ordinal:
            subset = subset.sort_values("setting", key=lambda c: c.astype(float),
                                        ascending=False)
        else:
            subset = subset.sort_values("f1_50", ascending=True)

        if stage in selected:
            highlight = subset["setting"] == selected[stage]
        else:
            highlight = subset["f1_50"] == subset["f1_50"].max()
        colours = [SERIES[0] if flag else "#a9c6ea" for flag in highlight]
        axis.barh(subset["setting"], subset["f1_50"], color=colours,
                  edgecolor=SURFACE, linewidth=1.2, height=0.68)
        chosen = selected.get(stage)
        for y, (setting, value) in enumerate(zip(subset["setting"], subset["f1_50"])):
            note = "  (selected on trainval)" if setting == chosen else ""
            axis.text(value + 0.012, y, f"{value:.3f}{note}", va="center",
                      fontsize=7.5, color=INK_SOFT)

        axis.set_xlim(0, 1.5 if stage in selected else 1.08)
        axis.set_title(titles[stage], color=INK, pad=8, loc="left", fontweight="bold")
        axis.set_xlabel("Detection F1 at IoU 0.5")
        axis.grid(axis="x", color=GRID, linewidth=0.7)
        axis.set_axisbelow(True)

    figure.suptitle(
        f"Detection F1 by preprocessing choice — {dataset.upper()}"
        f"{' (test split)' if dataset == 'deeppcb' else ''}",
        color=INK, fontweight="bold", x=0.005, ha="left", y=1.04,
    )
    _save(figure, f"fig_pipeline_f1_{dataset}.png")


def figure_rectify() -> None:
    """Rectification error against the dataset's own recorded angles."""
    frame = _read("rectify_per_board.csv")
    if frame is None:
        return
    frame = frame[frame["detected"]]

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    figure.subplots_adjust(wspace=0.32)

    full = frame[frame["scale"] == 1.0]
    axes[0].scatter(full["true_deg"], full["estimated_deg"], s=16,
                    color=SERIES[0], alpha=0.75, edgecolor=SURFACE, linewidth=0.5)
    limit = max(abs(full["true_deg"]).max(), abs(full["estimated_deg"]).max()) + 1.5
    axes[0].plot([-limit, limit], [-limit, limit], color=INK_SOFT,
                 linewidth=1.0, linestyle="--", zorder=0)
    axes[0].text(-limit + 0.4, limit - 2.2, "perfect estimate", fontsize=7.5, color=INK_SOFT)
    axes[0].set_xlabel("Angle applied by the dataset (degrees)")
    _style(axes[0], "Estimated against true rotation\n(full resolution, boards detected)",
           "Estimated angle (degrees)")

    for index, scale in enumerate(sorted(frame["scale"].unique())):
        errors = frame[frame["scale"] == scale]["relative_error_deg"]
        axes[1].hist(errors, bins=18, range=(0, 3), alpha=0.72,
                     color=SERIES[index], label=f"scale {scale:g}",
                     edgecolor=SURFACE, linewidth=0.8)
    axes[1].set_xlabel("Absolute angular error (degrees)")
    _style(axes[1], "Error distribution by\nestimation resolution", "Boards")
    axes[1].legend(frameon=False, fontsize=8)

    _save(figure, "fig_rectify_error.png")


def main() -> None:
    print("Writing figures to outputs/figures/")
    figure_denoise()
    figure_pipeline("deeppcb")
    figure_pipeline("hripcb")
    figure_rectify()


if __name__ == "__main__":
    main()
