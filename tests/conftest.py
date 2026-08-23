import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_CANDIDATES = ["out/sev_medium/drop/91731", "out/work/drop/91731"]
RELEASE_CANDIDATES = ["out/release_medium", "out/release"]


def find_workdir():
    for c in WORK_CANDIDATES:
        p = os.path.join(REPO, c)
        if os.path.exists(os.path.join(p, "plan.json")):
            return p
    return None


def find_release():
    import glob
    for c in RELEASE_CANDIDATES:
        p = os.path.join(REPO, c)
        if glob.glob(os.path.join(p, "clips", "**", "meta.json"), recursive=True):
            return p
    return None
