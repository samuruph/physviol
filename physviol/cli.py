"""physviol command line.

    conda activate physviol
    python -m physviol.cli taxonomy
    python -m physviol.cli generate --debug -n 2
    python -m physviol.cli annotate out/work/drop/0173
    python -m physviol.cli overlay out/release/clips/.../invalid_solidity_a
    python -m physviol.cli validate out/release

`generate` is the end-to-end path: it shells out to docker/kubric.sh for the
simulate+render half (container) and then runs annotation and overlays here
(host). The two halves meet at the trajectory seam.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from typing import List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- taxonomy
def cmd_taxonomy(a) -> int:
    from .taxonomy import (COMPATIBILITY, DOMAINS, FAMILIES, MEDIA, SCENARIOS,
                           SEVERITY_BINS, build_cells, validate_taxonomy)
    validate_taxonomy()
    print("MEDIA (%d)  -- level 0, the LikePhys-style macro-category" % len(MEDIA))
    for m, why in MEDIA.items():
        staged = sorted(n for n, v in SCENARIOS.items() if v.physics_medium == m)
        print("  %-11s %-58s %s" % (m, why, ", ".join(staged) or "(none staged)"))
    print()
    print("DOMAINS (%d)" % len(DOMAINS))
    for d, q in DOMAINS.items():
        fams = [f for f, v in FAMILIES.items() if v.domain == d]
        print("  %-12s %s" % (d, q))
        print("               families: %s" % ", ".join(fams))
    print("\nSCENARIOS (%d)" % len(SCENARIOS))
    for s, v in SCENARIOS.items():
        print("  %-16s %-46s [%s]" % (s, v.description, v.physics_medium))
    cells = build_cells()
    print("\nBUILD cells: %d" % len(cells))
    if a.verbose:
        for scen, fam in cells:
            print("  %-16s x %s" % (scen, fam))
    _print_release_size(cells, a)
    return 0


#: Measured wall-clock per rendered clip, including annotation and the overlay
#: video. tier v0 is 4x the pixels and ~2x the frames of the debug tier; L1's HDRI
#: environment costs about 5.5x an L0 render.
#: Wall clock per clip, from the per-frame renders measured in CLAUDE.md --
#: 1.75 s at 256sq and 7.16 s at 512sq, all seven passes -- times the tier's
#: frame count, and times ~4.6 for L1's HDRI environment. Both published tiers
#: are 512sq now, so both are priced off the same 7.16 s.
SECONDS_PER_CLIP = {("debug", "L0"): 8.0, ("debug", "L1"): 44.0,
                    ("v0", "L0"): 637.0, ("v0", "L1"): 2930.0,
                    ("v1", "L0"): 695.0, ("v1", "L1"): 3195.0}


def _print_release_size(cells, a) -> None:
    """What a given configuration would actually produce, and how long it takes.

    The question "how many clips is this" has a non-obvious answer, because a
    valid twin is shared by every family and severity staged on the same
    scenario and seed. Counting cells times bins times variants overstates the
    render cost by about a third.
    """
    from .taxonomy import SCENARIOS, SEVERITY_BINS
    n_bins = len(SEVERITY_BINS) if a.severity == "all" else 1
    variants = max(1, a.variants)
    scenarios = {s for s, _ in cells}

    invalid = len(cells) * n_bins * variants
    valid = len(scenarios) * variants          # one per scenario+seed, shared
    renders = invalid + valid
    rate = SECONDS_PER_CLIP.get((a.tier, a.complexity), 60.0)
    serial = renders * rate

    print("\n-- a release at tier %s / %s / severity %s / %d variant(s)"
          % (a.tier, a.complexity, a.severity, variants))
    print("   %d cells x %d bin(s) x %d variant(s) = %d invalid clips"
          % (len(cells), n_bins, variants, invalid))
    print("   + %d valid twins (one per scenario+seed, shared across families"
          "\n     and bins because the prefix is bit-identical) = %d renders"
          % (valid, renders))
    print("   ~%.1f h serial, ~%.1f h at the measured 1.92x on four workers"
          % (serial / 3600.0, serial / 3600.0 / 1.92))
    print("   media: %s" % ", ".join(
        "%s %d" % (m, sum(1 for s, _ in cells
                          if SCENARIOS[s].physics_medium == m))
        for m in sorted({SCENARIOS[s].physics_medium for s, _ in cells})))


# ---------------------------------------------------------------- generate
def cmd_generate(a) -> int:
    from .scenarios import TIERS
    from . import scenarios as scen_mod
    from .taxonomy import build_cells

    tier = "debug" if a.debug else a.tier
    if tier not in TIERS:
        from .scenarios.base import LEGACY_TIER_NAMES
        hint = LEGACY_TIER_NAMES.get(tier)
        print("unknown tier %r%s; valid tiers: %s"
              % (tier, (" -- renamed to %r" % hint) if hint else "",
                 ", ".join(TIERS)), file=sys.stderr)
        return 2

    # Individual dials, for sweeping one knob without inventing a tier.
    try:
        tier_obj = TIERS[tier].override(
            resolution=a.resolution, fps=a.fps, num_frames=a.frames,
            samples_per_pixel=a.spp)
    except ValueError as exc:
        print("bad tier override: %s" % exc, file=sys.stderr)
        return 2
    overrides = [("--resolution", a.resolution), ("--fps", a.fps),
                 ("--frames", a.frames), ("--spp", a.spp)]
    overrides = [(k, v) for k, v in overrides if v is not None]
    if overrides:
        print("tier %s with %s -> %s" % (
            tier, " ".join("%s %s" % kv for kv in overrides), tier_obj.name))

    from . import injectors
    have, inj = set(scen_mod.available()), set(injectors.available())
    cells = [c for c in build_cells() if c[0] in have and c[1] in inj]
    if a.scenario:
        cells = [c for c in cells if c[0] == a.scenario]
    if a.family:
        cells = [c for c in cells if c[1] == a.family]
    if a.scenario and a.family and not cells:
        cells = [(a.scenario, a.family)]        # force an off-matrix probe
    if a.limit:
        cells = cells[:a.limit]
    if not cells:
        print("no (scenario, family) cells selected", file=sys.stderr)
        return 2

    # One worker run per (scenario, seed) covering every family of that
    # scenario: they share a single scene build, a single HDRI load and a
    # single valid render. At complexity L1 the environment costs ~4.6x an L0
    # render, so re-paying it per family was most of the wall clock.
    by_scenario = {}
    for scenario, family in cells:
        by_scenario.setdefault(scenario, []).append(family)

    work = a.workdir or os.path.join("out", "work")
    rel = a.outdir or os.path.join("out", "release")

    done, failed, t0 = [], [], time.perf_counter()
    for v in range(a.variants):
        for scenario, families in sorted(by_scenario.items()):
            seed = a.seed + v
            rc, info = _run_worker(scenario, seed, tier, ",".join(families),
                                   a.severity, work, complexity=a.complexity,
                                   window=a.window,
                                   dials={"resolution": a.resolution,
                                          "fps": a.fps, "frames": a.frames,
                                          "spp": a.spp})
            if rc != 0:
                print("worker failed for %s/%d: %s"
                      % (scenario, seed, str(info)[:400]), file=sys.stderr)
                failed.append((scenario, seed, "worker"))
                if not a.keep_going:
                    return rc
                continue
            for bad in [x for x in info.get("variants", []) if not x.get("ok")]:
                failed.append((scenario, bad.get("family"), bad.get("error")))
                print("  !! %-15s %-17s %-6s %s"
                      % (scenario, bad.get("family"), bad.get("severity"),
                         bad.get("error")), file=sys.stderr)
            produced = [x["dir"] for x in info.get("variants", []) if x.get("ok")]
            for res in _annotate(info["outdir"], rel,
                                 overlay=not a.no_overlay, only=produced):
                done.append(res)
                print("  %-15s seed=%-5d %-17s %-6s t_event=%-3d lag=%-2d "
                      "vwin=%-13s sev=%.2f"
                      % (scenario, seed, res["family"], res["severity"],
                         res["t_event"], res["observability_lag"],
                         str(res["violation_windows"])[:13],
                         res.get("peak_severity", res["peak_score"])))
    dt = time.perf_counter() - t0
    print("\n%d pairs in %.1fs (%.1fs/pair)  ->  %s"
          % (len(done), dt, dt / max(len(done), 1), rel))
    if failed:
        print("%d cell(s) produced nothing:" % len(failed), file=sys.stderr)
        for row in failed:
            print("   %s" % (row,), file=sys.stderr)
    return 0


def _run_worker(scenario, seed, tier, family, severity, workdir,
                complexity="L0", window=None, dials=None):
    cmd = ["bash", os.path.join(REPO, "docker", "kubric.sh"),
           "physviol/render/worker.py", "--scenario", scenario,
           "--seed", str(seed), "--tier", tier, "--family", family,
           "--severity", severity, "--complexity", complexity,
           "--outdir", workdir]
    if window:
        cmd += ["--window", str(window)]
    for flag, value in (dials or {}).items():
        if value is not None:
            cmd += ["--%s" % flag, str(value)]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("PHASE0 "):
            info = json.loads(line[len("PHASE0 "):])
            return (0, info) if info.get("ok") else (3, info)
    return (p.returncode or 4, {"stderr": p.stderr[-600:]})


def _annotate(workdir, outroot, overlay=True, only=None):
    from .annotate.pipeline import annotate_work
    results = annotate_work(workdir, outroot, only=only)
    if overlay:
        from .viz.overlay import build
        for r in results:
            r["overlay"] = build(r["clips"]["invalid"])["path"]
    return results


def cmd_annotate(a) -> int:
    res = _annotate(a.workdir, a.outdir, overlay=not a.no_overlay)
    print(json.dumps(res, indent=2, default=str))
    return 0


def cmd_overlay(a) -> int:
    from .viz.overlay import build
    print(json.dumps(build(a.clip_dir, a.out, upscale=a.upscale), default=str))
    return 0


# ---------------------------------------------------------------- validate
def cmd_grid(a) -> int:
    from .viz.grid import build
    views = [v.strip() for v in a.views.split(",")] if a.views else None
    print(json.dumps(build(a.pair_dir, a.family, a.out, cell=a.cell,
                           views=views), default=str))
    return 0


def cmd_sheet(a) -> int:
    from .viz.grid import sheet
    print(json.dumps(sheet(a.pair_dir, a.out, view=a.view, cell=a.cell),
                     default=str))
    return 0


def cmd_coverage(a) -> int:
    from .viz.grid import coverage
    print(json.dumps(coverage(a.root, a.out), default=str))
    return 0


def cmd_config_path(a) -> int:
    """Where a config would write. Lets a script chain generate -> validate ->
    videos without hardcoding a path the config already knows.

    Reads the config's **generate** block, not its own: `outdir` is a setting of
    the run, and this command exists only to report it.
    """
    from . import config

    outdir = getattr(a, "outdir", None)
    if outdir is None and getattr(a, "config", None):
        _, subs = _build()
        valid = {ac.dest for ac in subs["generate"]._actions} - {"help", "config"}
        try:
            outdir = config.load(a.config, "generate", valid).get("outdir")
        except config.ConfigError as exc:
            print("config error: %s" % exc, file=sys.stderr)
            return 2
    print(outdir or "out/release")
    return 0


def cmd_validate(a) -> int:
    from .schema.validate import validate_release
    rep = validate_release(a.root)
    print(json.dumps(rep, indent=2))
    return 0 if rep["ok"] else 1


# ---------------------------------------------------------------------- #
def _build(suppress: bool = False):
    """The parser, and its subparsers by name.

    Built twice: once normally, and once with `argument_default=SUPPRESS` so
    the second pass reports *only* the options actually typed. That is what
    lets a config fill in the rest without ever overriding a flag -- comparing
    against the defaults instead would make "the user typed the default value"
    indistinguishable from "the user typed nothing".
    """
    kw = {"argument_default": argparse.SUPPRESS} if suppress else {}
    ap = argparse.ArgumentParser(prog="physviol", **kw)
    sub = ap.add_subparsers(dest="cmd", required=True)
    subs = {}

    def add_parser(name, **extra):
        q = sub.add_parser(name, **dict(extra, **kw))
        q.add_argument("--config", metavar="NAME|PATH",
                       help="settings file; a bare name resolves to "
                            "configs/<name>.yaml. Flags override it.")
        subs[name] = q
        return q

    p = add_parser("taxonomy",
                       help="print the taxonomy, and size a release")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="list every (scenario, family) cell")
    p.add_argument("--tier", default="v0", help="debug | v0 | v1")
    p.add_argument("--complexity", default="L0")
    p.add_argument("--severity", default="all")
    p.add_argument("--variants", type=int, default=5)
    p.set_defaults(fn=cmd_taxonomy)

    p = add_parser("generate", help="simulate+render+annotate end to end")
    p.add_argument("--debug", action="store_true", help="the debug tier (fast, unpublished)")
    p.add_argument("--tier", default="debug",
                   help="debug | v0 | v1 -- see `physviol taxonomy`")
    p.add_argument("--variants", type=int, default=1,
                   help="randomisations per cell: each is a fresh seed, so "
                        "sizes, speeds, colours, camera, HDRI and (where "
                        "physically neutral) the actor's shape all differ")
    p.add_argument("-n", "--limit", type=int, default=0,
                   help="cap the number of cells, for a quick smoke run")
    p.add_argument("--seed", type=int, default=91731)
    p.add_argument("--scenario", help="restrict to one scenario")
    p.add_argument("--family", help="restrict to one family")
    p.add_argument("--keep-going", action="store_true",
                   help="carry on past a failing cell and list them at the end")
    p.add_argument("--severity", default="strong",
                   help="weak|medium|strong, or 'all' for the whole ladder in "
                        "one run (default: strong -- one clean variant per "
                        "cell, which is what you want while checking coverage)")
    p.add_argument("--window", type=int, default=None,
                   help="uniform violation duration in frames (sustained "
                        "families only; instant ones stay 1 frame)")
    p.add_argument("--resolution", type=int,
                   help="override the tier's square render size, e.g. 128")
    p.add_argument("--fps", type=int, help="override the tier's frame rate")
    p.add_argument("--frames", type=int,
                   help="override the tier's clip length; must be 4k+1 "
                        "(13, 17, 21, 25, 29, ...) for VAE latent alignment")
    p.add_argument("--spp", type=int,
                   help="override the tier's samples per pixel (render noise "
                        "vs time; frame time is ~1.29 + 0.0074*spp at 256sq)")
    p.add_argument("--complexity", default="L1",
                   help="L0 solid bg .. L4 MOVi-F (see `taxonomy`)")
    p.add_argument("--workdir")
    p.add_argument("--outdir")
    p.add_argument("--no-overlay", action="store_true")
    p.set_defaults(fn=cmd_generate)

    p = add_parser("annotate", help="host-side annotation of a worker dir")
    p.add_argument("workdir")
    p.add_argument("--outdir", default="out/release")
    p.add_argument("--no-overlay", action="store_true")
    p.set_defaults(fn=cmd_annotate)

    p = add_parser("overlay", help="annotated mp4 for one invalid clip")
    p.add_argument("clip_dir")
    p.add_argument("--out")
    p.add_argument("--upscale", type=int, default=4)
    p.set_defaults(fn=cmd_overlay)

    p = add_parser("grid", help="one family: valid vs every severity, all views")
    p.add_argument("pair_dir", help=".../clips/<release>/<scenario>/<seed>/")
    p.add_argument("--family")
    p.add_argument("--out")
    p.add_argument("--views", help="comma list of rgb,mask,sev,causal,div,energy "
                                   "(default: every view the clips have)")
    p.add_argument("--cell", type=int, default=224)
    p.set_defaults(fn=cmd_grid)

    p = add_parser("sheet",
                   help="one scenario+seed: every family x every severity")
    p.add_argument("pair_dir", help=".../clips/<release>/<scenario>/<seed>/")
    p.add_argument("--view", default="mask",
                   choices=["rgb", "mask", "sev", "causal", "div", "energy"],
                   help="which annotation view fills every cell")
    p.add_argument("--out")
    p.add_argument("--cell", type=int)
    p.set_defaults(fn=cmd_sheet)

    p = add_parser("coverage",
                       help="one video tiling every invalid clip in a release")
    p.add_argument("root", nargs="?", default="out/release")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_coverage)

    p = add_parser("config-path",
                   help="print the outdir a config resolves to")
    p.add_argument("--outdir")
    p.set_defaults(fn=cmd_config_path)

    p = add_parser("validate", help="schema + cross-checks over a release")
    p.add_argument("root", default="out/release", nargs="?")
    p.set_defaults(fn=cmd_validate)

    if suppress:
        # `argument_default=SUPPRESS` is overridden by any explicit `default=`
        # on an individual argument, which is most of them -- so the first
        # version of this reported `--complexity` as typed on every run and a
        # config could never set it. Force it onto every action instead.
        for parser in [ap] + list(subs.values()):
            parser._defaults = {}
            for action in parser._actions:
                action.default = argparse.SUPPRESS
    return ap, subs


def main(argv: Optional[List[str]] = None) -> int:
    from . import config

    argv = sys.argv[1:] if argv is None else list(argv)
    ap, subs = _build()
    a = ap.parse_args(argv)
    typed = set(vars(_build(suppress=True)[0].parse_args(argv)))

    valid = {ac.dest for ac in subs[a.cmd]._actions} - {"help", "config"}
    try:
        settings = config.load(getattr(a, "config", None), a.cmd, valid)
    except config.ConfigError as exc:
        print("config error: %s" % exc, file=sys.stderr)
        return 2
    for key, value in settings.items():
        if key not in typed:
            setattr(a, key, value)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
