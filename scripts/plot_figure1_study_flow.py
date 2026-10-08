"""Render Figure 1: cohort assembly and temporal-holdout construction.

Source data are aggregate counts only and are stored in
06_Tables_Source_Data/figure1_study_flow_source.csv. No patient-level data are
read or exported. The outputs are a vector SVG/PDF plus 600-dpi PNG/TIFF files.

Usage
-----
python plot_figure1_study_flow.py --package-root <submission-package-root>
"""

from __future__ import annotations

import argparse
import csv
from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


# Radiology: Artificial Intelligence submission geometry: double-column width
# (180 mm) with ≥7-pt final lettering and editable SVG/PDF text.
MM_PER_INCH = 25.4
FIG_WIDTH = 180 / MM_PER_INCH
FIG_HEIGHT = 115 / MM_PER_INCH

PALETTE = {
    "blue": "#0072B2",       # development / model-building
    "blue_fill": "#EAF3F8",
    "orange": "#E69F00",     # temporal holdout / untouched evaluation
    "orange_fill": "#FFF4DE",
    "slate": "#3D4B5C",      # neutral cohort/era structure
    "slate_fill": "#F2F5F7",
    "line": "#64748B",
    "muted": "#5F6B7A",
    "footer": "#F7F8FA",
    "footer_edge": "#CBD5E1",
    "white": "#FFFFFF",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }
)


def read_source(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["node"]: row for row in csv.DictReader(handle)}
    expected = {
        "Screened cohort",
        "Excluded cases",
        "Eligible cohort",
        "2021 archive",
        "2023-2026 institutional cohort",
        "Development cohort",
        "Temporal held-out cohort",
    }
    missing = expected.difference(rows)
    if missing:
        raise ValueError(f"Missing Figure 1 source rows: {sorted(missing)}")
    return rows


def card(ax, x, y, width, height, title, lines, edge, fill, title_color="#1F2937"):
    """Draw a consistently typeset rounded card in data coordinates."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.055,rounding_size=0.14",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.55,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.20,
        y + height - 0.28,
        title,
        ha="left",
        va="top",
        fontsize=9.6,
        fontweight="bold",
        color=title_color,
        zorder=3,
    )
    line_y = y + height - 0.78
    for index, line in enumerate(lines):
        ax.text(
            x + 0.20,
            line_y - index * 0.31,
            line,
            ha="left",
            va="top",
            fontsize=7.7,
            color="#253347",
            zorder=3,
        )


def arrow(ax, start, end, *, color=PALETTE["line"], connectionstyle="arc3", zorder=1):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        shrinkA=1.0,
        shrinkB=1.0,
        linewidth=1.25,
        color=color,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    ax.add_patch(patch)


def route_label(ax, x, y, text, *, ha="center"):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=6.9,
        color=PALETTE["muted"],
        bbox={"boxstyle": "round,pad=0.16", "facecolor": PALETTE["white"], "edgecolor": "none"},
        zorder=4,
    )


def save_flattened_raster(fig, destination: Path, dpi: int) -> None:
    """Write an opaque white RGB raster without leaving an on-disk temporary file."""
    memory = BytesIO()
    fig.savefig(memory, format=destination.suffix.lstrip("."), dpi=dpi, facecolor=PALETTE["white"], transparent=False)
    memory.seek(0)
    with Image.open(memory) as image:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, PALETTE["white"])
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        save_kwargs = {"dpi": (dpi, dpi)}
        if destination.suffix.lower() in {".tif", ".tiff"}:
            save_kwargs["compression"] = "tiff_lzw"
        canvas.save(destination, **save_kwargs)
        canvas.close()
        rgba.close()
    memory.close()


def render(package_root: Path) -> list[Path]:
    source = package_root / "06_Tables_Source_Data" / "figure1_study_flow_source.csv"
    output_dir = package_root / "04_Main_Figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = read_source(source)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.patch.set_facecolor(PALETTE["white"])
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 8.65)
    ax.axis("off")

    ax.text(
        0.55,
        8.25,
        "Cohort assembly and temporal holdout",
        fontsize=10.2,
        fontweight="bold",
        va="top",
        color="#1F2937",
    )
    ax.text(
        0.55,
        7.89,
        "Patient-level allocation; no temporal held-out case contributed to model fitting or threshold selection.",
        fontsize=7.2,
        va="top",
        color=PALETTE["muted"],
    )

    overall = data["Eligible cohort"]
    screened = data["Screened cohort"]
    excluded = data["Excluded cases"]
    era_2021 = data["2021 archive"]
    era_late = data["2023-2026 institutional cohort"]
    development = data["Development cohort"]
    holdout = data["Temporal held-out cohort"]

    card(ax, 0.55, 5.00, 2.72, 1.80, "Initially screened", [
        f"n = {screened['total']} cases",
        "2021: 325  •  2023–2026: 708",
        f"Excluded after criteria: {excluded['total']}",
    ], PALETTE["slate"], PALETTE["slate_fill"])
    card(ax, 3.95, 5.00, 2.72, 1.80, "Eligible cohort", [
        f"n = {overall['total']} genotype-confirmed cases",
        "96 EGFR 19-del  •  20 EGFR 20-ins",
        "132 EGFR 21-L858R",
    ], PALETTE["slate"], PALETTE["slate_fill"])

    card(
        ax,
        7.15,
        6.06,
        2.62,
        1.30,
        "2021 archive",
        [f"n = {era_2021['total']}  •  {era_2021['egfr_20_ins']} EGFR 20-ins"],
        PALETTE["slate"],
        PALETTE["slate_fill"],
    )
    card(
        ax,
        7.15,
        3.60,
        2.62,
        1.30,
        "2023–2026 cohort",
        [f"n = {era_late['total']}  •  {era_late['egfr_20_ins']} EGFR 20-ins", "Accrual-order split"],
        PALETTE["slate"],
        PALETTE["slate_fill"],
    )

    card(
        ax,
        10.00,
        5.84,
        3.45,
        1.72,
        "Development cohort",
        [
            f"n = {development['total']}  •  {development['egfr_20_ins']} EGFR 20-ins",
            "77 cases from 2021",
            "+ 110 earliest cases from 2023–2026",
        ],
        PALETTE["blue"],
        PALETTE["blue_fill"],
    )
    card(
        ax,
        10.00,
        3.22,
        3.45,
        1.72,
        "Temporal held-out cohort",
        [
            f"n = {holdout['total']}  •  {holdout['egfr_20_ins']} EGFR 20-ins",
            "61 most recent cases from 2023–2026",
        ],
        PALETTE["orange"],
        PALETTE["orange_fill"],
    )

    # Screening cascade, then split by acquisition era.
    arrow(ax, (3.27, 5.90), (3.95, 5.90))
    ax.plot([6.67, 6.95], [5.90, 5.90], color=PALETTE["line"], linewidth=1.25, zorder=1)
    ax.plot([6.95, 6.95], [4.25, 6.71], color=PALETTE["line"], linewidth=1.25, zorder=1)
    arrow(ax, (6.95, 6.71), (7.15, 6.71))
    arrow(ax, (6.95, 4.25), (7.15, 4.25))

    # The 2021 archive flows completely into development.
    arrow(ax, (9.77, 6.71), (10.00, 6.70))

    # Late-era cohort is partitioned into early development and recent temporal holdout cases.
    ax.plot([9.77, 10.10], [4.25, 4.25], color=PALETTE["line"], linewidth=1.25, zorder=1)
    arrow(ax, (10.10, 4.25), (10.00, 6.44), connectionstyle="arc3,rad=0.0")
    arrow(ax, (10.10, 4.25), (10.00, 4.08), connectionstyle="arc3,rad=0.0")

    # Lower protocol banner gives the flow diagram its methodological message.
    footer = FancyBboxPatch(
        (0.55, 0.42),
        12.90,
        1.56,
        boxstyle="round,pad=0.055,rounding_size=0.12",
        facecolor=PALETTE["footer"],
        edgecolor=PALETTE["footer_edge"],
        linewidth=1.0,
        zorder=0,
    )
    ax.add_patch(footer)
    ax.text(0.80, 1.66, "Locked temporal evaluation", fontsize=8.8, fontweight="bold", va="top", color="#1F2937")
    ax.text(
        0.80,
        1.28,
        "Feature selection, repeated 5-fold cross-validation, sigmoid calibration, and recall-constrained\n"
        "threshold selection were performed in the development cohort only.",
        fontsize=7.25,
        va="top",
        color="#253347",
    )
    ax.text(
        0.80,
        0.83,
        "The temporal held-out cohort remained untouched until final evaluation.",
        fontsize=7.25,
        va="top",
        color="#253347",
        fontweight="normal",
    )

    fig.subplots_adjust(left=0.02, right=0.985, top=0.97, bottom=0.055)
    paths = []
    stem = output_dir / "Figure1_revised"
    for extension in [".svg", ".pdf"]:
        output = stem.with_suffix(extension)
        fig.savefig(output, facecolor=PALETTE["white"], transparent=False)
        paths.append(output)
    for extension in [".png", ".tiff"]:
        output = stem.with_suffix(extension)
        save_flattened_raster(fig, output, dpi=600)
        paths.append(output)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Root of the submission package (default: inferred from this script).",
    )
    args = parser.parse_args()
    for output in render(args.package_root.resolve()):
        print(output)


if __name__ == "__main__":
    main()
