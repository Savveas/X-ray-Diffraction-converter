# -*- coding: utf-8 -*-
"""
xy_to_pdf.py  -  Convert XRD .xy files (2Theta / Intensity) to PDF plots.

Works in Spyder (press F5) or from a terminal:  python xy_to_pdf.py file1.xy file2.xy

How to choose the files:
  1. Fill in INPUT_FILES below, OR
  2. Leave it empty and a file-selection dialog will open, OR
  3. Pass the files as command-line arguments.

For every input file "name.xy" a "name.pdf" is written next to it
(or into OUTPUT_DIR if you set it).

The PDF contains:
  - the header metadata (operator, anode, scan type, time per step, ...)
  - the pattern on a linear intensity axis
  - the same pattern on a log intensity axis (shows weak peaks/background)
  - a summary: number of points, 2theta range, step size, max peak position
"""

import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # draw straight to file; no window needed
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ----------------------------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------------------------
INPUT_FILES = [
    # r"C:\path\to\2thetaomega_004_28389_exported.xy",
]
OUTPUT_DIR = None              # None = same folder as the input file
PEAK_LABELS = True             # label the strongest peaks on the plot
N_PEAK_LABELS = 5              # how many peaks to label
PEAK_MIN_FRACTION = 0.01       # peaks must be > this fraction of the max
LINE_COLOR = "#1f4e9c"
# ----------------------------------------------------------------------------


def read_xy(path):
    """Read an .xy file. Returns (x, y, metadata_dict, raw_header_text).

    Lines that cannot be read as two numbers are treated as header lines.
    Works with space, tab, comma or semicolon separators.
    """
    header_lines = []
    xs, ys = [], []
    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = re.split(r"[,\s;]+", s)
            try:
                x, y = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                header_lines.append(s)
                continue
            xs.append(x)
            ys.append(y)

    if not xs:
        raise ValueError("No numeric data found in %s" % path)

    header = " ".join(header_lines).lstrip("'# ")
    # Parse  Key: "Value"  pairs  (e.g.  Anode: "Cu")
    meta = {k.strip(): v.strip() for k, v in re.findall(r'([\w ]+?):\s*"([^"]*)"', header)}
    return np.array(xs), np.array(ys), meta, header


def find_peaks_simple(x, y, n=5, min_fraction=0.01):
    """Return indices of the n highest local maxima (no scipy needed)."""
    if len(y) < 3:
        return []
    is_max = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
    idx = np.where(is_max)[0] + 1
    idx = idx[y[idx] >= min_fraction * y.max()]
    idx = idx[np.argsort(y[idx])[::-1]]
    chosen = []
    min_sep = (x.max() - x.min()) * 0.01          # keep labels apart
    for i in idx:
        if all(abs(x[i] - x[j]) > min_sep for j in chosen):
            chosen.append(i)
        if len(chosen) >= n:
            break
    return chosen


def make_pdf(xy_path, out_dir=None):
    x, y, meta, header = read_xy(xy_path)

    base = os.path.splitext(os.path.basename(xy_path))[0]
    out_dir = out_dir or os.path.dirname(os.path.abspath(xy_path))
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, base + ".pdf")

    step = np.median(np.diff(x)) if len(x) > 1 else float("nan")
    i_max = int(np.argmax(y))
    peaks = find_peaks_simple(x, y, N_PEAK_LABELS, PEAK_MIN_FRACTION) if PEAK_LABELS else []

    fig = plt.figure(figsize=(8.27, 11.69))          # A4 portrait
    fig.suptitle(base, fontsize=14, fontweight="bold", y=0.975)

    # ---- metadata / summary text block ----
    ax_t = fig.add_axes([0.08, 0.84, 0.84, 0.10])
    ax_t.axis("off")
    left = ["Source file: " + os.path.basename(xy_path)]
    if meta:
        left += ["%s: %s" % (k, v) for k, v in meta.items() if v != ""]
    elif header:
        left.append(header[:120])
    right = [
        "Points: %d" % len(x),
        "2θ range: %.4f – %.4f °" % (x.min(), x.max()),
        "Step size: %.4f °" % step,
        "Max intensity: %.4g at 2θ = %.4f °" % (y[i_max], x[i_max]),
    ]
    ax_t.text(0.0, 1.0, "\n".join(left), va="top", fontsize=9, family="monospace")
    ax_t.text(0.52, 1.0, "\n".join(right), va="top", fontsize=9, family="monospace")

    def draw(ax, log):
        ax.plot(x, y, color=LINE_COLOR, lw=0.8)
        ax.set_xlabel("2θ (°)")
        ax.set_ylabel("Intensity (counts)")
        ax.set_xlim(x.min(), x.max())
        ax.grid(True, alpha=0.3, which="both")
        if log:
            ax.set_yscale("log")
            pos = y[y > 0]
            ax.set_ylim(max(pos.min() * 0.5, 1e-2) if len(pos) else 1e-2, y.max() * 3)
            ax.set_title("Log intensity scale", fontsize=10)
        else:
            ax.set_ylim(0, y.max() * 1.12)
            ax.set_title("Linear intensity scale", fontsize=10)
        for i in peaks:
            ax.annotate("%.3f°" % x[i], (x[i], y[i]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7, color="crimson")

    ax1 = fig.add_axes([0.10, 0.47, 0.84, 0.32])
    draw(ax1, log=False)
    ax2 = fig.add_axes([0.10, 0.07, 0.84, 0.32])
    draw(ax2, log=True)

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        d = pdf.infodict()
        d["Title"] = base
        d["Subject"] = "XRD pattern converted from " + os.path.basename(xy_path)
    plt.close(fig)
    return pdf_path


def choose_files():
    """File-selection dialog (falls back to typing a path)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        files = filedialog.askopenfilenames(
            title="Select .xy file(s)",
            filetypes=[("XY files", "*.xy *.xye *.txt *.dat"), ("All files", "*.*")])
        root.destroy()
        return list(files)
    except Exception:
        p = input("Path to .xy file (several: separate by ';'): ")
        return [s.strip().strip('"') for s in p.split(";") if s.strip()]


def main():
    files = list(INPUT_FILES)
    if not files:
        argv = [a for a in sys.argv[1:] if os.path.exists(a)]   # Spyder adds odd args
        files = argv or choose_files()
    if not files:
        print("No files selected.")
        return
    for f in files:
        try:
            out = make_pdf(f, OUTPUT_DIR)
            print("OK   %s  ->  %s" % (f, out))
        except Exception as e:
            print("FAIL %s : %s" % (f, e))


if __name__ == "__main__":
    main()
