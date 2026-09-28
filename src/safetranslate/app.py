"""Setia: a web app that runs the translation workflow and shows what it did.

Usage: uv run --group app python -m safetranslate.app configs/workflow.yaml [--preview]

--preview shows the UI with a canned example instead of calling models (no GPU needed).

The officer pastes English, picks a language, and sees the translation, a summary, a
diagram of the workflow, and a step-by-step walkthrough of every sentence: each attempt,
the critical errors found (highlighted by category), retries, and escalations.
"""

import html
import sys

from safetranslate.data.schema import ErrorItem

APP_NAME = "Setia"
SUBTITLE = "Critical Error Detection for Machine Translation in Singapore's Official Languages"
LANGUAGES = {"Chinese (中文)": "zh", "Malay (Melayu)": "ms", "Tamil (தமிழ்)": "ta"}
# One muted colour per critical-error category, readable on white
CATEGORY_COLOURS = {
    "wrong_quantity": "#f8c9c4",
    "wrong_name": "#fbd9a8",
    "flipped_meaning": "#dcc8f0",
    "added_information": "#c6dcf5",
    "removed_information": "#d9dde3",
}
CATEGORY_LABELS = {
    "wrong_quantity": "Wrong quantity",
    "wrong_name": "Wrong name",
    "flipped_meaning": "Flipped meaning",
    "added_information": "Added information",
    "removed_information": "Removed information",
}
# Civic blue: navy primary on white and slate, as agreed for a government-tool feel
CSS = """
.setia-header {background: #1f3a5f; color: white; padding: 18px 22px; border-radius: 8px;}
.setia-header h1 {margin: 0; font-size: 28px; color: white;}
.setia-header p {margin: 4px 0 0; color: #d6e0ee;}
.setia-flow {font-size: 16px; color: #1e293b; background: #f1f5f9; border-radius: 8px; padding: 12px 16px; line-height: 2;}
.setia-flow span {background: #eef2f7; color: #1e293b; border: 1px solid #cbd5e1; border-radius: 6px; padding: 2px 8px;}
.setia-sentence {border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; margin: 8px 0; background: white; color: #1e293b;}
.setia-badge {font-weight: 600; border-radius: 12px; padding: 1px 10px; font-size: 13px;}
.setia-published {background: #dcfce7; color: #166534;}
.setia-fixed {background: #fef3c7; color: #92400e;}
.setia-review {background: #fee2e2; color: #991b1b;}
.setia-step {margin: 4px 0 4px 8px; color: #334155;}
mark {border-radius: 3px; padding: 0 2px; color: #111827;}
"""

def theme():
    """Civic blue theme with a neutral font (Inter), falling back to system fonts."""
    import gradio as gr

    font = [gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
    civic = gr.themes.Soft(primary_hue="blue", neutral_hue="slate", font=font)
    # The look is white with a navy header; copy every light colour into dark mode so
    # browsers set to dark mode show exactly the same page
    for name in [n for n in vars(civic) if n.endswith("_dark")]:
        setattr(civic, name, getattr(civic, name.removesuffix("_dark")))
    return civic


FLOW_HTML = (
    '<div class="setia-flow">How each sentence is handled: <span>① Translate</span> → '
    "<span>② Check for critical errors</span> → no errors: <span>✅ Publish</span> · "
    "errors: <span>🔁 Retry with the errors as feedback (up to 3 attempts)</span> · "
    "still wrong after 3: <span>⚠️ Human review</span></div>"
)


def highlight(text: str, errors: list[ErrorItem]) -> str:
    """HTML for a translation with each error's quoted text highlighted by category."""
    marked = html.escape(text)
    for error in errors:
        if error.category == "removed_information" or not error.span:
            continue
        span = html.escape(error.span)
        colour = CATEGORY_COLOURS[error.category]
        title = html.escape(f"{CATEGORY_LABELS[error.category]}: {error.description}")
        marked = marked.replace(span, f'<mark style="background:{colour}" title="{title}">{span}</mark>', 1)
    return marked


def legend() -> str:
    items = " ".join(
        f'<mark style="background:{colour}">{CATEGORY_LABELS[c]}</mark>' for c, colour in CATEGORY_COLOURS.items()
    )
    return f"<div>Error colours: {items}</div>"


def badge(state: dict) -> str:
    if state["status"] == "escalated":
        return '<span class="setia-badge setia-review">⚠️ Needs human review</span>'
    if state["attempts"] > 1:
        return f'<span class="setia-badge setia-fixed">🔁 Fixed on attempt {state["attempts"]}</span>'
    return '<span class="setia-badge setia-published">✅ Published</span>'


def walkthrough(state: dict) -> str:
    """Step-by-step HTML of what the workflow did with one sentence."""
    steps = []
    for number, attempt in enumerate(state["history"], start=1):
        errors = [ErrorItem(**e) for e in attempt["errors"]]
        retry_note = " (with feedback on the previous errors)" if number > 1 else ""
        steps.append(f'<div class="setia-step">① Attempt {number} translated{retry_note}: {highlight(attempt["translation"], errors)}</div>')
        if not errors:
            steps.append('<div class="setia-step">② Checked: no critical errors</div>')
            continue
        found = "; ".join(
            f"{CATEGORY_LABELS[e.category]}: {html.escape(e.description)}"
            + (f" (missing: “{html.escape(e.span)}”)" if e.category == "removed_information" else "")
            for e in errors
        )
        steps.append(f'<div class="setia-step">② Checked: found {found}</div>')
    ending = {
        "published": "✅ Published",
        "escalated": f"⚠️ Sent for human review after {state['attempts']} attempts",
    }[state["status"]]
    steps.append(f'<div class="setia-step"><b>{ending}</b></div>')
    return (
        f'<div class="setia-sentence">{badge(state)} <b>{html.escape(state["source"])}</b>'
        + "".join(steps)
        + "</div>"
    )


def summary(states: list[dict]) -> str:
    total = len(states)
    first = sum(s["status"] == "published" and s["attempts"] == 1 for s in states)
    fixed = sum(s["status"] == "published" and s["attempts"] > 1 for s in states)
    review = sum(s["status"] == "escalated" for s in states)
    return (
        f"**{total} sentences** · ✅ {first} published first time · "
        f"🔁 {fixed} fixed by retry · ⚠️ {review} need human review"
    )


def build_app(run_document):
    """Builds the Gradio UI around a function (text, lang) -> translate_document result."""
    import gradio as gr

    def on_translate(text: str, language: str):
        if not text.strip():
            return "Paste some English text first.", "", ""
        result = run_document(text, LANGUAGES[language])
        states = result["segments"]
        return summary(states), result["translation"], "".join(walkthrough(s) for s in states)

    with gr.Blocks(title=APP_NAME) as app:
        gr.HTML(f'<div class="setia-header"><h1>{APP_NAME}</h1><p>{SUBTITLE}</p></div>')
        gr.HTML(FLOW_HTML)
        # Same size side by side (like a translation tool); controls below both
        with gr.Row(equal_height=True):
            source = gr.Textbox(label="English text", lines=12, max_lines=12, placeholder="Paste the notice to translate…")
            # Editable, so an officer can correct sentences sent for human review
            translation = gr.Textbox(label="Translation (you can correct it here)", lines=12, max_lines=12)
        with gr.Row():
            language = gr.Dropdown(list(LANGUAGES), value=list(LANGUAGES)[1], label="Translate into", scale=1)
            button = gr.Button("Translate and check", variant="primary", scale=1)
        overview = gr.Markdown()
        gr.HTML(legend())
        gr.Markdown("### What the workflow did, sentence by sentence")
        details = gr.HTML()
        button.click(on_translate, [source, language], [overview, translation, details])
    return app


def preview_document(text: str, lang: str) -> dict:
    """A canned result for reviewing the UI without running any models."""
    error = ErrorItem(category="wrong_quantity", span="500 g", description="Source says 500 mg; translation says 500 g")
    flip = ErrorItem(category="flipped_meaning", span="dengan alkohol", description="Source says do not take with alcohol")
    fixed = {
        "source": "Take 500 mg of paracetamol twice a day.", "status": "published", "attempts": 2,
        "translation": "Ambil 500 mg parasetamol dua kali sehari.",
        "history": [
            {"translation": "Ambil 500 g parasetamol dua kali sehari.", "errors": [error.model_dump()]},
            {"translation": "Ambil 500 mg parasetamol dua kali sehari.", "errors": []},
        ],
    }
    escalated = {
        "source": "Do not take with alcohol.", "status": "escalated", "attempts": 3,
        "translation": "Ambil dengan alkohol.",
        "history": [{"translation": "Ambil dengan alkohol.", "errors": [flip.model_dump()]}] * 3,
    }
    clean = {
        "source": "See a doctor if the fever lasts more than three days.", "status": "published",
        "attempts": 1, "translation": "Jumpa doktor jika demam berlarutan lebih daripada tiga hari.",
        "history": [{"translation": "Jumpa doktor jika demam berlarutan lebih daripada tiga hari.", "errors": []}],
    }
    states = [fixed, escalated, clean]
    return {"translation": " ".join(s["translation"] for s in states), "segments": states}


def main() -> None:
    import gradio as gr

    if "--preview" in sys.argv:
        build_app(preview_document).launch(
            theme=theme(), css=CSS
        )
        return

    from safetranslate.alter.generate import make_client
    from safetranslate.config import load_harness_config, load_workflow_config
    from safetranslate.workflow.graph import build_graph, translate_document

    config = load_workflow_config(sys.argv[1])
    harness = load_harness_config(config.harness_config)
    translator, evaluator = harness.systems[config.translator], harness.systems[config.evaluator]
    graph = build_graph(
        (make_client(translator), translator), (make_client(evaluator), evaluator), config.max_attempts
    )
    app = build_app(lambda text, lang: translate_document(graph, text, lang))
    app.launch(theme=theme(), css=CSS)


if __name__ == "__main__":
    main()
