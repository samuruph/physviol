"""Comparison grid -- the real clip beside every violation level, annotated.

Columns are variants (valid, then each severity in order); rows are views
(RGB, RGB+mask, severity map). Each column carries its own red ACTIVE dot and
its own window bar, because the severity bins have different windows.

    +-------------------------------------------------------------+
    | scenario / family                                    f 7/12  |
    +----------------+----------------+----------------+----------+
    | VALID          | easy      [o]  | medium    [o]  | hard [o] |
    | RGB            | RGB            | RGB            | RGB      |
    +----------------+----------------+----------------+----------+
    | (no mask)      | + MASK         | + MASK         | + MASK   |
    +----------------+----------------+----------------+----------+
    | (no severity)  | SEVERITY       | SEVERITY       | SEVERITY |
    +----------------+----------------+----------------+----------+
    | window bars, one per column                                 |
    +-------------------------------------------------------------+

mp4 only -- no image files are written.
"""
from __future__ import annotations

import glob
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

ORDER = {"valid": 0, "weak": 1, "medium": 2, "strong": 3}


def build(pair_dir: str, family: Optional[str] = None,
          out_path: Optional[str] = None, cell: int = CELL) -> Dict[str, object]:
    """`pair_dir` is .../clips/<release>/<scenario>/<seed>/ -- the folder holding
    the valid clip and its invalid variants."""
    import cv2

    cols = _collect(pair_dir, family)
    if len(cols) < 2:
        raise ValueError("need a valid clip and at least one variant in %s" % pair_dir)

    T = int(cols[0]["meta"]["num_frames"])
    n = len(cols)
    rows = 3
    W = n * cell + (n + 1) * PAD
    H = HEADER + COLHEAD + rows * (cell + PAD) + PAD + BAR
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

        for i, c in enumerate(cols):
            x = PAD + i * (cell + PAD)
            y = HEADER + COLHEAD
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

            for r, kind in enumerate(("rgb", "mask", "sev")):
                yy = y + r * (cell + PAD)
                if c["is_valid"] and kind == "sev":
                    cv2.rectangle(f, (x, yy), (x + cell - 1, yy + cell - 1),
                                  (44, 44, 52), 1)
                    ov._text(f, "n/a (valid clip)", (x + 8, yy + cell // 2),
                             ov.C_DIM, 0.40, 1)
                    continue
                # The valid column shows its reference mask -- the lawful
                # footprint every invalid variant departed from.
                img = ov._panel(kind, t, c["rgb"],
                                None if c["is_valid"] else c["mask"],
                                c["sev"], None, None, cell, c["ref"])
                f[yy:yy + cell, x:x + cell] = img
                cv2.rectangle(f, (x, yy), (x + cell - 1, yy + cell - 1),
                              (60, 60, 70), 1)
                if kind == "mask":
                    src = c["ref"] if c["is_valid"] else c["mask"]
                    if src is not None:
                        ov._text(f, "%s %d px" % ("should-be" if c["is_valid"]
                                                  else "violation",
                                                  int(src[t].sum())),
                                 (x + 5, yy + cell - 7), ov.C_DIM, 0.36, 1)
                if kind == "sev" and c["tl"] is not None:
                    ov._text(f, "s=%.3f" % float(c["tl"]["severity_t"][t]),
                             (x + 5, yy + cell - 7), (255, 255, 255), 0.40, 1)

            # -- per-column window bar --------------------------------------
            by = H - BAR + 4
            cv2.rectangle(f, (x, by), (x + cell - 1, by + 7), (40, 40, 48), -1)
            for s, e in c["vwin"]:
                cv2.rectangle(f, (x + int(s / T * cell), by),
                              (x + max(int((e + 1) / T * cell) - 1,
                                       int(s / T * cell) + 1), by + 7),
                              ov.C_MASK, -1)
            for s, e in c["owin"]:
                cv2.rectangle(f, (x + int(s / T * cell), by + 8),
                              (x + max(int((e + 1) / T * cell) - 1,
                                       int(s / T * cell) + 1), by + 13),
                              ov.C_OBS, -1)
            px = x + min(cell - 1, int((t + 0.5) / T * cell))
            cv2.line(f, (px, by - 2), (px, by + 14), (255, 255, 255), 1)
            if c["lag"] is not None:
                ov._text(f, "lag %d" % c["lag"], (x + 4, H - 3), ov.C_DIM, 0.34, 1)

        out[t] = f

    out_path = out_path or os.path.join(pair_dir, "grid_%s.mp4" % (fam or "all"))
    vid.write(out, out_path, fps=int(meta0.get("fps", 12)))
    return {"path": out_path, "columns": [c["label"] for c in cols], "frames": T}


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
            "tl": np.load(tlp) if os.path.exists(tlp) else None,
            "vwin": [tuple(w) for w in v.get("violation_windows", [])],
            "owin": [tuple(w) for w in v.get("observable_windows", [])],
            "lag": v.get("observability_lag_frames"),
            "mag": (v.get("intervention") or {}).get("magnitude"),
            "mag_unit": (v.get("intervention") or {}).get("magnitude_unit", ""),
        })
    return sorted(cols, key=lambda c: c["sort"])


def coverage(release_root: str, out_path: Optional[str] = None,
             cell: int = 200, cols: int = 3) -> Dict[str, object]:
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

    T = min(int(c["meta"]["num_frames"]) for c in clips)
    rows = (len(clips) + cols - 1) // cols
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
