"""The demo backend."""
import pytest

from tests.synthetic import DRY, WEED


def test_finding_confidence_uses_only_its_tokens():
    import math
    from app.backend.engine import findings
    toks = [("Large", -0.1), (" drydown", -0.1), (" in", -0.1), (" the", -0.1), (" centre", -0.1), (",", -2.0),
            (" with", -2.0), (" possibly", -1.0), (" small", -1.0), (" weed", -1.0), (" cluster", -1.0),
            (" at", -1.0), (" the", -1.0), (" top", -1.0), (" left", -1.0), (".", -3.0)]
    caption = "".join(t for t, _ in toks)
    rows = findings(caption, toks, 0.5)
    assert [r["cls"] for r in rows] == [DRY, WEED]
    assert rows[0]["conf"] == pytest.approx(math.exp(-0.1), abs=1e-3)
    assert rows[1]["conf"] == pytest.approx(math.exp(-1.0), abs=1e-3) and rows[1]["possibly"] and rows[1]["hedge"]
    assert rows[0]["extent"] == "large" and rows[0]["loc"] == "centre"
