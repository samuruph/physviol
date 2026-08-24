"""Two comparison videos over a generated pair directory.

`build` -- **one family, every severity.** Columns are variants (valid, then
each bin in order); rows are the same annotation views the single-clip overlay
shows, so a grid cell and an overlay panel are directly comparable. Answers "how
do weak, medium and strong differ, and is each one annotated correctly".

    +--------------------------------------------------------------------+
    | drop / solidity                    seed 777  tier debug     f 7/24  |
    +---------+----------------+----------------+----------------+-------+
    |         | VALID          | weak     [o]   | medium   [o]   | strong|
    | RGB     |                |                |                |       |
    +---------+----------------+----------------+----------------+-------+
    | MASK    | (should-be)    | + MASK         | + MASK         |       |
    +---------+----------------+----------------+----------------+-------+
    | SEVERITY| n/a            | heat + scale   | heat + scale   |       |
    +---------+----------------+----------------+----------------+-------+
    | CAUSAL  | ...            |                |                |       |
    +---------+----------------+----------------+----------------+-------+
    | DIVERG. | ...            |                |                |       |
    +---------+----------------+----------------+----------------+-------+

`sheet` -- **every family at once**, for one scenario and seed. Rows are
families, columns are valid plus each severity bin, one annotation view
throughout (`view=`). Answers "do all of these violations look like what they
claim, side by side" -- which is the question a per-family video cannot answer
because you have to remember the last one.

mp4 only -- no image files are written.
"""
from __future__ import annotations

import glob
import math
import json
import os
from typing import Dict, List, Optional

import numpy as np

from . import overlay as ov
from . import video as vid

CELL = 224
PAD = 4
HEADER = 30
COLHEAD = 20
BAR = 26
#: Left-hand gutter for row labels. Labelling rows in the gutter rather than
#: inside the panels keeps the imagery unobstructed -- the panels are the thing
#: being judged, and text drawn over a mask is text drawn over the evidence.
GUTTER = 104

ORDER = {"valid": 0, "weak": 1, "medium": 2, "strong": 3}

#: (row label, `overlay._panel` kind, which array must exist). The labels match
#: the single-clip overlay's panel titles on purpose, so the two videos can be
#: read against each other without translating.
VIEWS = [
    ("RGB", "rgb", None),
    ("MASK", "mask", "mask"),
    ("SEVERITY", "sev", "sev"),
    ("CAUSAL", "causal", "causal"),
    ("DIVERGENCE", "div", "div"),
    ("ENERGY", "energy", "energy"),
]
VIEW_NOTE = {
    "mask": "red = violation   green = should-be",
    "sev": "0..1, inferno, gamma for display",
    "causal": "1 = culprit   2 = affected",
    "div": "|valid - invalid|  NOT ground truth",
    "energy": "per-body E, fraction of E0",
}


def build(pair_dir: str, family: Optional[str] = None,
          out_path: Optional[str] = None, cell: int = CELL,
          views: Optional[List[str]] = None) -> Dict[str, object]:
    """One family, every severity bin, every annotation view.

    `pair_dir` is .../clips/<release>/<scenario>/<seed>/ -- the folder holding
    the valid clip and its invalid variants.
    """
    import cv2

    cols = _collect(pair_dir, family)
    if len(cols) < 2:
        raise ValueError("need a valid clip and at least one variant in %s" % pair_dir)

    rows = _rows_for(cols, views)
    T = int(cols[0]["meta"]["num_frames"])
    n = len(cols)
    W = GUTTER + n * cell + (n + 1) * PAD
    H = HEADER + COLHEAD + len(rows) * (cell + PAD) + PAD + BAR
    out = np.zeros((T, H, W, 3), np.uint8)

    meta0 = cols[0]["meta"]
    fam = family or next((c["meta"].get("family") for c in cols[1:]), "-")

    for t in range(T):
        f = np.full((H, W, 3), ov.C_BG, np.uint8)
        title = "%s / %s   |   seed %s  tier %s  %s" % (
            meta0.get("scenario"), fam, meta0.get("seed"), meta0.get("tier"),
            (meta0.get("complexity") or {}).get("name", ""))
        ov._text(f, title, (PAD + 2, 20), ov.C_TEXT, 0.50, 1)
        fs = "f %d/%d" % (t, T - 1)
        ov._text(f, fs, (W - PAD - ov._w(fs, 0.46), 20), ov.C_TEXT, 0.46, 1)

        for r, (label, kind, _need) in enumerate(rows):
            yy = HEADER + COLHEAD + r * (cell + PAD)
            ov._text(f, label, (PAD + 2, yy + 16), (255, 255, 255), 0.44, 1)
            note = VIEW_NOTE.get(kind)
            if note:
                for k, part in enumerate(_wrap(note, 15)):
                    ov._text(f, part, (PAD + 2, yy + 32 + 12 * k),
                             ov.C_DIM, 0.32, 1)

        for i, c in enumerate(cols):
            x = GUTTER + PAD + i * (cell + PAD)
            active = bool(c["tl"]["active"][t]) if c["tl"] is not None else False

            # -- column header: label, magnitude, live ACTIVE dot -----------
            lab = c["label"]
            ov._text(f, lab, (x + 4, HEADER + 14), ov.C_TEXT, 0.44, 1)
            if c["mag"] is not None:
                sub = "%.3g %s" % (c["mag"], c["mag_unit"].replace("m_", ""))
                ov._text(f, sub, (x + 4 + ov._w(lab, 0.44) + 8, HEADER + 14),
                         ov.C_DIM, 0.36, 1)
            if active:
                cv2.circle(f, (x + cell - 10, HEADER + 9), 6, ov.C_ACTIVE, -1)
                cv2.circle(f, (x + cell - 10, HEADER + 9), 6, (255, 255, 255), 1)

            for r, (_label, kind, need) in enumerate(rows):
                yy = HEADER + COLHEAD + r * (cell + PAD)
                _draw_cell(f, c, kind, need, t, x, yy, cell)

            # -- per-column window bar --------------------------------------
            by = H - BAR + 4
            _window_bar(f, c, t, T, x, by, cell)
            if c["lag"] is not None:
                ov._text(f, "lag %d" % c["lag"], (x + 4, H - 3), ov.C_DIM, 0.34, 1)
        out[t] = f

    out_path = out_path or os.path.join(pair_dir, "grid_%s.mp4" % (fam or "all"))
    vid.write(out, out_path, fps=int(meta0.get("fps", 12)))
    return {"path": out_path, "columns": [c["label"] for c in cols],
            "rows": [r[0] for r in rows], "frames": T}


def sheet(pair_dir: str, out_path: Optional[str] = None,
          view: str = "mask", cell: Optional[int] = None) -> Dict[str, object]:
    """Every family of one scenario+seed at once: rows families, columns bins.

    The view a per-family grid cannot give you. Judging whether `solidity` and
    `permanence` are really different violations means seeing them at the same
    moment of the same scene, not remembering the last video.
    """
    import cv2

    cols = _collect(pair_dir, None)
    valid = [c for c in cols if c["is_valid"]]
    if not valid:
        raise ValueError("no valid clip in %s" % pair_dir)
    by_family: Dict[str, Dict[str, Dict]] = {}
    for c in cols:
        if c["is_valid"]:
            continue
        by_family.setdefault(c["meta"].get("family", "?"), {})[c["label"]] = c
    if not by_family:
        raise ValueError("no invalid clips in %s" % pair_dir)

    bins = [b for b in ("weak", "medium", "strong")
            if any(b in v for v in by_family.values())]
    families = sorted(by_family)
    ncol = 1 + len(bins)
    nrow = len(families)
    if cell is None:
        cell = 200 if nrow <= 8 else (160 if nrow <= 14 else 128)

    T = int(valid[0]["meta"]["num_frames"])
    W = GUTTER + ncol * cell + (ncol + 1) * PAD
    H = HEADER + COLHEAD + nrow * (cell + PAD) + PAD
    out = np.zeros((T, H, W, 3), np.uint8)
    meta0 = valid[0]["meta"]
    need = {"rgb": None, "mask": "mask", "sev": "sev",
            "causal": "causal", "div": "div", "energy": "energy"}[view]

    for t in range(T):
        f = np.full((H, W, 3), ov.C_BG, np.uint8)
        title = "%s  seed %s  %s  %s  |  %s" % (
            meta0.get("scenario"), meta0.get("seed"), meta0.get("tier"),
            (meta0.get("complexity") or {}).get("name", ""),
            {k: l for l, k, _ in VIEWS}.get(view, view))
        ov._text(f, title, (PAD + 2, 20), ov.C_TEXT, 0.50, 1)
        fs = "f %d/%d" % (t, T - 1)
        ov._text(f, fs, (W - PAD - ov._w(fs, 0.46), 20), ov.C_TEXT, 0.46, 1)
        # The legend only goes on the title line if it actually fits between
        # the title and the frame counter; otherwise it goes under the gutter
        # heading, where there is always room. Text that runs off the edge is
        # worse than text somewhere less obvious.
        note = VIEW_NOTE.get(view)
        if note:
            nx = PAD + 4 + ov._w(title, 0.50) + 14
            if nx + ov._w(note, 0.38) < W - PAD - ov._w(fs, 0.46) - 10:
                ov._text(f, note, (nx, 20), ov.C_DIM, 0.38, 1)
            else:
                ov._text(f, note, (PAD + 2, HEADER + 14), ov.C_DIM, 0.34, 1)

        for j, lab in enumerate(["VALID"] + bins):
            x = GUTTER + PAD + j * (cell + PAD)
            ov._text(f, lab, (x + 4, HEADER + 14), ov.C_TEXT, 0.46, 1)

        for i, famname in enumerate(families):
            yy = HEADER + COLHEAD + i * (cell + PAD)
            ov._text(f, famname[:16], (PAD + 2, yy + 15), (255, 255, 255),
                     0.40, 1)
            any_col = next(iter(by_family[famname].values()))
            dom = any_col["meta"].get("domain") or ""
            if dom:
                ov._text(f, dom[:16], (PAD + 2, yy + 29), ov.C_DIM, 0.33, 1)

            for j, lab in enumerate(["VALID"] + bins):
                x = GUTTER + PAD + j * (cell + PAD)
                c = valid[0] if lab == "VALID" else by_family[famname].get(lab)
                if c is None:
                    cv2.rectangle(f, (x, yy), (x + cell - 1, yy + cell - 1),
                                  (44, 44, 52), 1)
                    ov._text(f, "not built", (x + 6, yy + cell // 2),
                             ov.C_DIM, 0.36, 1)
                    continue
                _draw_cell(f, c, view, need, t, x, yy, cell, compact=True)
                if not c["is_valid"] and c["tl"] is not None \
                        and bool(c["tl"]["active"][t]):
                    cv2.circle(f, (x + cell - 9, yy + 9), 5, ov.C_ACTIVE, -1)
                    cv2.circle(f, (x + cell - 9, yy + 9), 5, (255, 255, 255), 1)
        out[t] = f

    out_path = out_path or os.path.join(pair_dir, "sheet_%s.mp4" % view)
    vid.write(out, out_path, fps=int(meta0.get("fps", 12)))
    return {"path": out_path, "families": families, "bins": bins,
            "view": view, "frames": T}


# ------------------------------------------------------------------ shared
def _rows_for(cols, views: Optional[List[str]]):
    """Which annotation rows this pair can actually fill.

    Driven by what is on disk, not by a fixed list: a release generated without
    divergence maps should lose the row rather than draw an empty box captioned
    with a promise the files do not keep.
    """
    have = set()
    for c in cols:
        for key in ("mask", "sev", "causal", "div", "energy"):
            if c.get(key) is not None:
                have.add(key)
    rows = [v for v in VIEWS if v[2] is None or v[2] in have]
    if views:
        rows = [v for v in rows if v[1] in views]
    return rows


def _draw_cell(f, c, kind, need, t, x, y, cell, compact: bool = False) -> None:
    import cv2
    if need is not None and c.get(need) is None and not c["is_valid"]:
        cv2.rectangle(f, (x, y), (x + cell - 1, y + cell - 1), (44, 44, 52), 1)
        ov._text(f, "no %s" % need, (x + 6, y + cell // 2), ov.C_DIM, 0.36, 1)
        return
    # A valid clip has no violation to shade, so severity/causal/divergence are
    # blank by definition rather than missing. Saying which keeps the column
    # honest -- an empty panel with no caption reads as a failed render.
    if c["is_valid"] and kind in ("sev", "causal", "div"):
        cv2.rectangle(f, (x, y), (x + cell - 1, y + cell - 1), (44, 44, 52), 1)
        ov._text(f, "n/a (valid clip)", (x + 6, y + cell // 2), ov.C_DIM, 0.38, 1)
        return

    img = ov._panel(kind, t, c["rgb"],
                    None if c["is_valid"] else c["mask"],
                    c["sev"], c["causal"], c["div"], cell, c["ref"],
                    c.get("energy"))
    f[y:y + cell, x:x + cell] = img
    cv2.rectangle(f, (x, y), (x + cell - 1, y + cell - 1), (60, 60, 70), 1)

    if kind == "mask":
        src = c["ref"] if c["is_valid"] else c["mask"]
        if src is not None:
            ov._text(f, "%s %d px" % ("should-be" if c["is_valid"] else "violation",
                                      int(src[t].sum())),
                     (x + 5, y + cell - 7), ov.C_DIM, 0.36, 1)
    elif kind == "energy" and c.get("etrace") is not None:
        tr = c["etrace"]
        if compact:
            ov._text(f, "E %.1fJ" % float(tr["total"][t]), (x + 5, y + cell - 7),
                     (255, 255, 255), 0.38, 1)
        else:
            ov._energy_curve(f, x, y, cell, t, tr)
    elif kind == "sev" and c["tl"] is not None:
        s = float(c["tl"]["severity_t"][t])
        if compact:
            ov._text(f, "s=%.3f" % s, (x + 5, y + cell - 7), (255, 255, 255),
                     0.40, 1)
        else:
            ov._sev_scale(f, x, y, cell, s)


def _window_bar(f, c, t, T, x, by, cell) -> None:
    import cv2
    cv2.rectangle(f, (x, by), (x + cell - 1, by + 7), (40, 40, 48), -1)
    for s, e in c["vwin"]:
        cv2.rectangle(f, (x + int(s / T * cell), by),
                      (x + max(int((e + 1) / T * cell) - 1,
                               int(s / T * cell) + 1), by + 7), ov.C_MASK, -1)
    for s, e in c["owin"]:
        cv2.rectangle(f, (x + int(s / T * cell), by + 8),
                      (x + max(int((e + 1) / T * cell) - 1,
                               int(s / T * cell) + 1), by + 13), ov.C_OBS, -1)
    px = x + min(cell - 1, int((t + 0.5) / T * cell))
    cv2.line(f, (px, by - 2), (px, by + 14), (255, 255, 255), 1)


def _wrap(s: str, width: int) -> List[str]:
    lines, cur = [], ""
    for word in s.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    if cur:
        lines.append(cur)
    return lines[:3]


def _collect(pair_dir: str, family: Optional[str]) -> List[Dict]:
    cols = []
    for mp in sorted(glob.glob(os.path.join(pair_dir, "*", "meta.json"))):
        cdir = os.path.dirname(mp)
        with open(mp) as fh:
            meta = json.load(fh)
        is_valid = meta.get("label") == "valid"
        if not is_valid and family and meta.get("family") != family:
            continue
        v = meta.get("violation") or {}
        sev = (v.get("intervention") or {}).get("severity_bin", "valid")
        tlp = os.path.join(cdir, "timelines.npz")
        cols.append({
            "dir": cdir, "meta": meta, "is_valid": is_valid,
            "label": "VALID" if is_valid else sev,
            "sort": ORDER.get("valid" if is_valid else sev, 9),
            "rgb": ov._rgb(cdir, int(meta["num_frames"])),
            "mask": ov._npz(cdir, "violation_mask.npz", "mask"),
            "ref": ov._npz(cdir, "reference_mask.npz", "mask"),
            "sev": (lambda a: None if a is None else a.astype(np.float32))(
                ov._npz(cdir, "severity_map.npz", "severity")),
            "causal": ov._npz(cdir, "causal_mask.npz", "mask"),
            "energy": (lambda a: None if a is None else a.astype(np.float32))(
                ov._npz(cdir, "energy_map.npz", "energy")),
            "etrace": ov._energy_trace(cdir),
            "div": (lambda a: None if a is None else a.astype(np.float32))(
                ov._npz(cdir, "divergence_map.npz", "divergence")),
            "tl": np.load(tlp) if os.path.exists(tlp) else None,
            "vwin": [tuple(w) for w in v.get("violation_windows", [])],
            "owin": [tuple(w) for w in v.get("observable_windows", [])],
            "lag": v.get("observability_lag_frames"),
            "mag": (v.get("intervention") or {}).get("magnitude"),
            "mag_unit": (v.get("intervention") or {}).get("magnitude_unit", ""),
        })
    return sorted(cols, key=lambda c: c["sort"])


def coverage(release_root: str, out_path: Optional[str] = None,
             cell: Optional[int] = None, cols: Optional[int] = None) -> Dict[str, object]:
    """One video tiling every invalid clip in a release: RGB with the violation
    mask and the reference outline, captioned with scenario / family / lag.

    The "is all of this working" view. Each tile is a different (scenario,
    family) cell, so a broken generator shows up as an obviously wrong tile
    rather than something you would only notice by opening 9 files.
    """
    import cv2

    clips = []
    for mp in sorted(glob.glob(os.path.join(release_root, "clips", "**",
                                            "meta.json"), recursive=True)):
        cdir = os.path.dirname(mp)
        with open(mp) as fh:
            meta = json.load(fh)
        if meta.get("label") != "invalid":
            continue
        v = meta.get("violation") or {}
        tlp = os.path.join(cdir, "timelines.npz")
        clips.append({
            "meta": meta,
            "rgb": ov._rgb(cdir, int(meta["num_frames"])),
            "mask": ov._npz(cdir, "violation_mask.npz", "mask"),
            "ref": ov._npz(cdir, "reference_mask.npz", "mask"),
            "tl": np.load(tlp) if os.path.exists(tlp) else None,
            "lag": v.get("observability_lag_frames", 0),
        })
    if not clips:
        raise ValueError("no invalid clips under %s" % release_root)

    # Lay out roughly square and shrink the tiles as the release grows. A
    # fixed three-column grid was fine for nine cells and produces a 3800px
    # column for forty-eight, which no player will show you all of at once.
    n = len(clips)
    if cols is None:
        cols = max(3, int(math.ceil(math.sqrt(n))))
    if cell is None:
        cell = 200 if n <= 16 else (160 if n <= 36 else 128)
    T = min(int(c["meta"]["num_frames"]) for c in clips)
    rows = (n + cols - 1) // cols
    cap = 30
    W = cols * cell + (cols + 1) * PAD
    H = HEADER + rows * (cell + cap + PAD) + PAD
    out = np.zeros((T, H, W, 3), np.uint8)

    for t in range(T):
        f = np.full((H, W, 3), ov.C_BG, np.uint8)
        ov._text(f, "coverage: %d cells   f %d/%d" % (len(clips), t, T - 1),
                 (PAD + 2, 20), ov.C_TEXT, 0.50, 1)
        for i, c in enumerate(clips):
            r, col = divmod(i, cols)
            x = PAD + col * (cell + PAD)
            y = HEADER + r * (cell + cap + PAD)
            img = ov._panel("mask", t, c["rgb"], c["mask"], None, None, None,
                            cell, c["ref"])
            f[y:y + cell, x:x + cell] = img
            cv2.rectangle(f, (x, y), (x + cell - 1, y + cell - 1), (60, 60, 70), 1)
            active = bool(c["tl"]["active"][t]) if c["tl"] is not None else False
            if active:
                cv2.circle(f, (x + cell - 11, y + 11), 6, ov.C_ACTIVE, -1)
                cv2.circle(f, (x + cell - 11, y + 11), 6, (255, 255, 255), 1)
            m = c["meta"]
            ov._text(f, "%s" % m.get("scenario"), (x + 3, y + cell + 12),
                     ov.C_TEXT, 0.40, 1)
            ov._text(f, "%s  lag=%d" % (m.get("family"), c["lag"]),
                     (x + 3, y + cell + 25), ov.C_DIM, 0.38, 1)
        out[t] = f

    out_path = out_path or os.path.join(release_root, "coverage.mp4")
    vid.write(out, out_path, fps=int(clips[0]["meta"].get("fps", 12)))
    return {"path": out_path, "cells": len(clips), "frames": T}
