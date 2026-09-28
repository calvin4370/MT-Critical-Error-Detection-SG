import re
from safetranslate.app import (
    EXAMPLE_ENGLISH,
    EXAMPLE_TRANSLATIONS,
    check_card,
    check_summary,
    highlight,
    preview_check,
    summary,
    walkthrough,
)
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


def test_check_card_highlights_errors_and_missing_english() -> None:
    wrong = ErrorItem(category="wrong_quantity", span="18", description="8 became 18").model_dump()
    missing = ErrorItem(category="removed_information", span="after meals", description="dropped").model_dump()
    state = {"source": "Take 8 after meals.", "translation": "Ambil 18.", "unit": "sentence",
             "history": [{"translation": "Ambil 18.", "errors": [wrong, missing]}]}
    card = check_card(state)
    assert "Critical errors found" in card
    assert '">18</mark>' in card and '">after meals</mark>' in card
    assert "Missing: “after meals”" in card


def test_check_summary_counts_and_warns_on_paragraph_mismatch() -> None:
    clean = {"history": [{"errors": []}]}
    flagged = {"history": [{"errors": [{"category": "wrong_name", "span": "x", "description": "d"}]}]}
    summary = check_summary({"segments": [clean, flagged, clean], "paragraphs": (2, 1)})
    text = re.sub(r"<[^>]+>", "", summary)
    assert "3 sentences checked · 1 with critical errors · 2 with no critical errors found" in text
    assert "not the intended use" in summary


def test_preview_check_flags_the_example_number_change() -> None:
    result = preview_check(EXAMPLE_ENGLISH, EXAMPLE_TRANSLATIONS["ms"], "ms")
    flagged = [s for s in result["segments"] if s["history"][0]["errors"]]
    assert [s["history"][0]["errors"][0]["span"] for s in flagged] == ["18"]
