from safetranslate.app import highlight, summary, walkthrough
from safetranslate.data.schema import ErrorItem

QUANTITY = ErrorItem(category="wrong_quantity", span="500 g", description="Source says 500 mg")
FIXED = {
    "source": "Take 500 mg.", "status": "published", "attempts": 2,
    "history": [
        {"translation": "Ambil 500 g.", "errors": [QUANTITY.model_dump()]},
        {"translation": "Ambil 500 mg.", "errors": []},
    ],
}


def test_highlight_marks_the_error_span_and_escapes_html() -> None:
    out = highlight("Ambil 500 g <b>.", [QUANTITY])
    assert "<mark" in out and "500 g</mark>" in out
    assert "&lt;b&gt;" in out  # user text can't inject HTML


def test_walkthrough_shows_each_attempt_and_the_fix() -> None:
    out = walkthrough(FIXED)
    assert "Fixed on attempt 2" in out
    assert "Attempt 2 translated (with feedback" in out
    assert "no critical errors" in out


def test_summary_counts_outcomes() -> None:
    escalated = {**FIXED, "status": "escalated", "attempts": 3}
    first_time = {**FIXED, "attempts": 1}
    assert summary([FIXED, escalated, first_time]) == (
        "**3 sentences** · ✅ 1 published first time · 🔁 1 fixed by retry · ⚠️ 1 need human review"
    )
