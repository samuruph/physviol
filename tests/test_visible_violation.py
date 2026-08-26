"""Every built cell must depict something a viewer can see.

Your rule, made checkable: "if some invalid doesn't make sense because it is not
changing anything or it is not a visible violation, let's just skip it for that
specific scenario". A cell that carries a full set of labels describing a
violation the video does not contain is worse than a cell that does not exist --
it teaches a model that nothing is something.

Runs against a generated release, so it skips without one. `physviol audit`
prints the same measurement in a readable form.
"""
from __future__ import annotations

import pytest

from conftest import find_release
from physviol.annotate.audit import audit, is_invisible


@pytest.fixture(scope="module")
def release():
    r = find_release()
    if r is None:
        pytest.skip("no release; run `bash scripts/run.sh`")
    return r


def test_no_built_cell_is_invisible(release):
    rows = audit(release)
    if not rows:
        pytest.skip("no invalid clips in the release")
    blind = [r for r in rows if is_invisible(r)]
    assert not blind, (
        "%d cell(s) depict nothing visible -- add them to "
        "taxonomy.NOT_MEANINGFUL or give the family something to act on:\n%s"
        % (len(blind), "\n".join(
            "   %s x %s (%s): severity %.3f, %d observable frames, "
            "evidence %.4f" % (r["scenario"], r["family"], r["severity_bin"],
                               r["peak_severity"], r["observable_frames"],
                               r["evidence"])
            for r in blind)))
