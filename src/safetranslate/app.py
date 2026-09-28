"""LinguaSG: a web app that checks translations for critical errors.

Usage: uv run --group app python -m safetranslate.app configs/workflow.yaml [--preview]

--preview shows the UI without calling models (no GPU needed): the translate tab shows a
canned example, and the check tab simulates the evaluator by comparing numbers.

Tab 1, "Check your translation": the officer pastes English and a translation from any MT
tool; each sentence pair is checked, with critical errors highlighted by category.
Tab 2, "Translate and check": the officer pastes English and sees the translation and a
step-by-step walkthrough of the workflow: attempts, errors found, retries, escalations.
"""

import html
import re
import sys

from safetranslate.data.schema import ErrorItem

APP_NAME = "LinguaSG"
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
# Apple-style product look: light grey page, white rounded cards, one blue accent
CSS = """
/* A fixed width: otherwise the page shrinks to its widest element and the views differ */
.gradio-container {width: min(1400px, 94vw) !important; max-width: min(1400px, 94vw) !important; margin: 0 auto !important;}
footer {display: none !important;}
/* The hero reserves space; the title and tagline are fixed and moved by the scroll script */
.setia-hero {height: 200px;}
.setia-title, .setia-tagline {position: fixed; top: 0; left: 0; margin: 0; transform-origin: 0 0; white-space: nowrap; z-index: 1001; pointer-events: none; opacity: 0; will-change: transform;}
.setia-title {font-size: 56px; font-weight: 700; letter-spacing: -0.03em; color: #1d1d1f; line-height: 1.2;}
.setia-tagline {font-size: 19px; line-height: 1.4; color: #6e6e73;}
.setia-bar {position: fixed; top: 0; left: 0; right: 0; height: 64px; z-index: 1000; pointer-events: none; opacity: 0;
    background: rgba(245,245,247,.72); backdrop-filter: saturate(180%) blur(20px); -webkit-backdrop-filter: saturate(180%) blur(20px);
    border-bottom: 1px solid rgba(0,0,0,.06);}
.setia-subtitle {text-align: center; font-size: 17px; line-height: 1.5; color: #6e6e73; max-width: 760px; margin: 4px auto 18px; text-wrap: balance;}
/* Two views switched by the segmented control; only the first shows at load */
#setia-view-0, #setia-view-1 {width: 100% !important; align-self: stretch !important;}
#setia-view-1 {display: none;}
@keyframes setia-fade {from {opacity: 0; transform: translateY(6px);} to {opacity: 1; transform: none;}}
.setia-view-in {animation: setia-fade .45s cubic-bezier(.32,.72,0,1);}
.setia-tabs {text-align: center; margin: 0 0 24px;}
/* Segmented control: one grey pill track; a black pill slides to the selected option */
.setia-seg {position: relative; display: inline-flex; background: #e8e8ed; border-radius: 999px; padding: 0;}
.setia-seg button {position: relative; z-index: 1; background: transparent; border: none; border-radius: 999px; padding: 9px 22px;
    font: inherit; font-size: 15px; font-weight: 500; color: #1d1d1f; cursor: pointer; transition: color .35s ease !important;}
.setia-seg button.on {color: #fff;}
.setia-seg-pill {position: absolute; top: 0; bottom: 0; left: 0; width: 0; border-radius: 999px; background: #1d1d1f;
    transition: left .5s cubic-bezier(.32,.72,0,1), width .5s cubic-bezier(.32,.72,0,1);}
.setia-field-label {font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; color: #86868b; margin: 0 0 6px 4px;}
.setia-hidden {display: none !important;}
/* Smooth, quiet motion */
button {transition: background-color .25s ease, transform .15s ease !important;}
button:active {transform: scale(.98);}
@keyframes setia-rise {from {opacity: 0; transform: translateY(12px);} to {opacity: 1; transform: none;}}
@keyframes setia-shimmer {from {background-position: 200% 0;} to {background-position: -200% 0;}}
.setia-shimmer {height: 14px; border-radius: 7px; margin: 10px 0; background: linear-gradient(90deg, #ececf0 25%, #f7f7f9 50%, #ececf0 75%);
    background-size: 200% 100%; animation: setia-shimmer 1.4s linear infinite;}
/* A faint blue glow behind the title */
body, gradio-app {background: radial-gradient(1100px 520px at 50% -160px, rgba(0,113,227,.14), rgba(0,113,227,0) 70%), #f5f5f7 !important;}
.setia-title.animate, .setia-tagline.animate {transition: transform .55s cubic-bezier(.32,.72,0,1);}
.setia-bar {transition: opacity .6s ease;}
.setia-input {position: relative;}
.setia-input:focus-within {box-shadow: inset 0 0 0 2px rgba(0,113,227,.35) !important; border-radius: 16px; transition: box-shadow .25s ease;}
.setia-counter {position: absolute; bottom: 10px; right: 18px; font-size: 12px; color: #86868b; pointer-events: none; font-variant-numeric: tabular-nums;}
/* DeepL-style: both panes in one white card, split by a thin divider */
.setia-panes {background: #fff; border-radius: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.05), 0 8px 24px rgba(0,0,0,.05); gap: 0 !important; padding: 6px; overflow: hidden;}
.setia-panes .form {border: none !important; background: transparent !important; box-shadow: none !important;}
.setia-panes .block {border-radius: 0 !important;}
.setia-panes .form > .block:first-child {border-right: 1px solid #e5e5ea !important;}
.setia-input .icon-button-wrapper, .setia-input .icon-button {border: none !important; background: transparent !important; box-shadow: none !important;}
.setia-input textarea {border: none !important; box-shadow: none !important; background: transparent !important; font-size: 17px !important; line-height: 1.5 !important; color: #1d1d1f !important;
    box-sizing: border-box !important; height: 320px !important; min-height: 320px !important; max-height: 320px !important; overflow-y: auto !important; resize: none !important;}
/* Placeholder centred in the empty box; on focus it disappears and the cursor is top left */
.setia-input textarea:placeholder-shown:not(:focus) {padding-top: 146px !important; text-align: center;}
.setia-input textarea:focus::placeholder {color: transparent;}
.setia-controls {align-items: flex-end !important; justify-content: center !important; flex-wrap: nowrap !important; gap: 12px !important; margin-top: 18px;}
.setia-controls > * {flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; margin: 0 !important;}
.setia-controls > .column {gap: 0 !important;}
/* The hidden language radio's empty wrapper would otherwise push the pills up */
.setia-controls .form:has(.setia-hidden) {display: none !important;}
.setia-controls .html-container, .setia-controls .prose {padding: 0 !important; margin: 0 !important;}
.setia-controls button:not(.setia-seg button) {padding: 0 26px !important; align-self: flex-end !important;}
.setia-controls > .form, .setia-controls .block {padding: 0 !important;}
.setia-controls button, .setia-seg button {height: 40px !important; min-height: 40px !important; max-height: 40px !important; padding-top: 0 !important; padding-bottom: 0 !important;}
/* Two fixed lines on wide screens; normal wrapping on narrow ones */
@media (min-width: 1000px) {.setia-subtitle-wide {max-width: none; white-space: nowrap;}}
.setia-results-title {font-size: 20px; font-weight: 600; color: #1d1d1f; margin: 28px 0 4px; letter-spacing: -0.01em;}
.setia-summary {font-size: 20px; font-weight: 600; color: #1d1d1f; margin: 28px 0 6px; letter-spacing: -0.01em;}
.setia-legend {display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 8px; color: #6e6e73; font-size: 14px; margin: 8px 0;}
.setia-legend .setia-chip {margin: 0;}
.setia-sentence {background: #fff; border-radius: 18px; padding: 18px 22px; margin: 14px 0; color: #1d1d1f; box-shadow: 0 1px 3px rgba(0,0,0,.05), 0 6px 20px rgba(0,0,0,.04);
    animation: setia-rise .6s cubic-bezier(.32,.72,0,1) backwards; transition: transform .35s cubic-bezier(.32,.72,0,1), box-shadow .35s ease;}
.setia-sentence:hover {transform: translateY(-2px); box-shadow: 0 2px 6px rgba(0,0,0,.06), 0 14px 32px rgba(0,0,0,.08);}
.setia-skeleton:hover {transform: none;}
mark {transition: box-shadow .2s ease;}
.setia-error {border-radius: 10px; padding: 2px 6px; margin-left: -6px; transition: background-color .2s ease;}
.setia-pair {display: grid; grid-template-columns: 1fr 1fr; gap: 28px; margin: 12px 0 4px;}
.setia-label {font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; color: #86868b; margin-bottom: 4px;}
.setia-text {font-size: 16px; line-height: 1.6;}
.setia-error {margin: 10px 0 0; font-size: 15px; color: #1d1d1f;}
.setia-chip {display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: 13px; font-weight: 600; margin-right: 8px; color: #1d1d1f;}
.setia-badge {font-weight: 600; border-radius: 999px; padding: 3px 12px; font-size: 13px;}
.setia-published {background: #e8f7ee; color: #1a7f37;}
.setia-fixed {background: #fff4e0; color: #9a5b00;}
.setia-review {background: #fdecec; color: #c9252c;}
.setia-step {margin: 6px 0 6px 2px; color: #1d1d1f; line-height: 1.6;}
.setia-warning {background: #fff4e0; color: #9a5b00; border-radius: 14px; padding: 12px 16px; margin-top: 10px; font-size: 15px; font-weight: 400;}
.setia-note {color: #86868b; font-size: 13px; margin: 8px 0 0;}
mark {border-radius: 4px; padding: 1px 3px; color: #1d1d1f;}
/* People who ask their system for less motion get none */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {animation: none !important; transition: none !important;}
}
"""
# Hovering an error's pill outlines its highlighted text, and vice versa (one rule per error)
CSS += "".join(
    f'.setia-sentence:has(.setia-error[data-err="{i}"]:hover) mark[data-err="{i}"] {{box-shadow: 0 0 0 2px #1d1d1f;}}'
    f'.setia-sentence:has(mark[data-err="{i}"]:hover) .setia-error[data-err="{i}"] {{background: #f5f5f7;}}'
    for i in range(10)
)
# Page behaviour: the title snaps between centre and header in one smooth move; segmented
# controls drive Gradio's hidden tabs and language choice; summary numbers count up; each
# pane shows a character counter. Gradio renders after load, so elements are watched for.
HEAD = """
<script>
(() => {
  const ease = t => 1 - Math.pow(1 - t, 3);
  let heroDone = false;
  function hero() {
    const title = document.querySelector('.setia-title'), tagline = document.querySelector('.setia-tagline');
    const bar = document.querySelector('.setia-bar'), spot = document.querySelector('.setia-hero');
    if (!title || !tagline || !bar || !spot) return;
    heroDone = true;
    // Fixed elements are moved to <body> so no container can trap their positioning
    [bar, title, tagline].forEach(el => document.body.appendChild(el));
    let collapsed = window.scrollY > 24;
    const apply = () => {
      const p = collapsed ? 1 : 0, width = document.documentElement.clientWidth;
      const top = spot.getBoundingClientRect().top + window.scrollY;
      const tw = title.offsetWidth, gw = tagline.offsetWidth;
      // In the header both sit on one line, centred on the bar's middle (32 px)
      const th = title.offsetHeight, gh = tagline.offsetHeight;
      const t = p ? [28, 32 - th * 0.4 / 2, 0.4] : [(width - tw) / 2, top + 36, 1];
      const g = p ? [28 + tw * 0.4 + 16, 32 - gh * 0.74 / 2, 0.74] : [(width - gw) / 2, top + 36 + th + 6, 1];
      title.style.transform = `translate(${t[0]}px, ${t[1]}px) scale(${t[2]})`;
      tagline.style.transform = `translate(${g[0]}px, ${g[1]}px) scale(${g[2]})`;
      bar.style.opacity = p;
      title.style.opacity = tagline.style.opacity = 1;
    };
    // Two states only; the gap between 4 and 24 px stops it flickering at the boundary
    window.addEventListener('scroll', () => {
      const next = window.scrollY > 24 ? true : window.scrollY < 4 ? false : collapsed;
      if (next !== collapsed) { collapsed = next; apply(); }
    }, {passive: true});
    window.addEventListener('resize', apply);
    apply();
    requestAnimationFrame(() => requestAnimationFrame(() => { title.classList.add('animate'); tagline.classList.add('animate'); }));
  }
  function segments() {
    document.querySelectorAll('.setia-seg:not([data-bound])').forEach(seg => {
      seg.dataset.bound = '1';
      const pill = seg.querySelector('.setia-seg-pill'), buttons = [...seg.querySelectorAll('button')];
      const place = () => {
        const on = seg.querySelector('button.on');
        if (!on || !on.offsetWidth) return;
        // The first placement is instant, so the pill doesn't slide in from the edge
        if (!pill.dataset.placed) pill.style.transition = 'none';
        pill.style.left = on.offsetLeft + 'px'; pill.style.width = on.offsetWidth + 'px';
        if (!pill.dataset.placed) { pill.offsetWidth; pill.style.transition = ''; pill.dataset.placed = '1'; }
      };
      buttons.forEach((button, i) => button.addEventListener('click', () => {
        buttons.forEach(b => b.classList.toggle('on', b === button));
        place();
        if (seg.dataset.kind === 'views') {
          buttons.forEach((_, j) => {
            const view = document.getElementById('setia-view-' + j);
            if (!view) return;
            view.style.display = j === i ? 'flex' : 'none';
            view.classList.toggle('setia-view-in', j === i);
          });
        } else {
          const radios = document.querySelectorAll('#' + seg.dataset.target + ' input[type=radio]');
          if (radios[i]) radios[i].click();
        }
      }));
      new ResizeObserver(place).observe(seg);
      place();
      // A link ending in #translate opens the second view
      if (seg.dataset.kind === 'views' && location.hash === '#translate' && buttons[1]) buttons[1].click();
    });
  }
  const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function countUp() {
    document.querySelectorAll('.setia-count:not([data-done])').forEach(el => {
      el.dataset.done = '1';
      if (still) return;
      const to = +el.dataset.to, start = performance.now();
      const step = now => {
        const t = Math.min((now - start) / 700, 1);
        el.textContent = Math.round(to * ease(t));
        if (t < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }
  function counters() {
    document.querySelectorAll('.setia-input:not(.setia-no-count) textarea:not([data-counted])').forEach(area => {
      area.dataset.counted = '1';
      const badge = document.createElement('div');
      badge.className = 'setia-counter';
      area.closest('.setia-input').appendChild(badge);
      const max = area.maxLength > 0 ? ' / ' + area.maxLength.toLocaleString() : '';
      const update = () => { badge.textContent = area.value.length.toLocaleString() + max; };
      area.addEventListener('input', update);
      // Clear and "Try an example" change the text without an input event
      setInterval(update, 400);
      update();
    });
  }
  const run = () => { if (!heroDone) hero(); segments(); countUp(); counters(); };
  new MutationObserver(run).observe(document.documentElement, {childList: true, subtree: true});
  run();
})();
</script>
"""
MAX_CHARACTERS = 5000
# "Try an example": a medicine notice whose translation has one critical error (8 -> 18)
EXAMPLE_ENGLISH = (
    "Take one tablet twice a day after meals.\n"
    "Do not take more than 8 tablets in 24 hours. See a doctor if the fever lasts more than three days."
)
EXAMPLE_TRANSLATIONS = {
    "zh": "每天两次，饭后服用一片。\n24小时内服用不得超过18片。如果发烧持续超过三天，请就医。",
    "ms": (
        "Ambil satu tablet dua kali sehari selepas makan.\n"
        "Jangan ambil lebih daripada 18 tablet dalam masa 24 jam. "
        "Jumpa doktor jika demam berlarutan lebih daripada tiga hari."
    ),
    "ta": (
        "உணவுக்குப் பிறகு ஒரு மாத்திரையை நாளொன்றுக்கு இரண்டு முறை எடுத்துக் கொள்ளுங்கள்.\n"
        "24 மணி நேரத்தில் 18 மாத்திரைகளுக்கு மேல் எடுத்துக் கொள்ள வேண்டாம். "
        "காய்ச்சல் மூன்று நாட்களுக்கு மேல் நீடித்தால் மருத்துவரைப் பார்க்கவும்."
    ),
}

def theme():
    """Apple-style light theme in Geist: grey page, white blocks, pill buttons, blue accent."""
    import gradio as gr

    font = [gr.themes.GoogleFont("Geist"), "ui-sans-serif", "system-ui", "sans-serif"]
    look = gr.themes.Base(primary_hue="blue", neutral_hue="gray", font=font).set(
        body_background_fill="#f5f5f7",
        body_text_color="#1d1d1f",
        block_background_fill="#ffffff",
        block_border_width="0px",
        block_shadow="none",
        block_radius="20px",
        block_label_text_color="#86868b",
        block_label_background_fill="transparent",
        input_background_fill="#ffffff",
        input_radius="14px",
        button_large_radius="999px",
        button_small_radius="999px",
        button_primary_background_fill="#0071e3",
        button_primary_background_fill_hover="#0077ed",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#e8e8ed",
        button_secondary_background_fill_hover="#dcdce1",
        button_secondary_text_color="#1d1d1f",
    )
    # Light only: copy every light colour into dark mode so dark-mode browsers match
    for name in [n for n in vars(look) if n.endswith("_dark")]:
        setattr(look, name, getattr(look, name.removesuffix("_dark")))
    return look




def highlight(text: str, errors: list[ErrorItem], missing: bool = False) -> str:
    """HTML for text with each error's quoted text highlighted by category.

    Args:
        missing: Highlight only removed-information errors, whose quote is from the English
            (they're absent from the translation); otherwise all other errors.
    """
    marked = html.escape(text)
    for number, error in enumerate(errors):
        if (error.category == "removed_information") != missing or not error.span:
            continue
        span = html.escape(error.span)
        colour = CATEGORY_COLOURS[error.category]
        title = html.escape(f"{CATEGORY_LABELS[error.category]}: {error.description}")
        marked = marked.replace(span, f'<mark data-err="{number}" style="background:{colour}" title="{title}">{span}</mark>', 1)
    return marked


def legend() -> str:
    items = "".join(
        f'<span class="setia-chip" style="background:{colour}">{CATEGORY_LABELS[c]}</span>' for c, colour in CATEGORY_COLOURS.items()
    )
    return f'<div class="setia-legend"><span>Error colours:</span>{items}</div>'


def badge(state: dict) -> str:
    if state["status"] == "escalated":
        return '<span class="setia-badge setia-review">⚠️ Needs human review</span>'
    if state["attempts"] > 1:
        return f'<span class="setia-badge setia-fixed">🔁 Fixed on attempt {state["attempts"]}</span>'
    return '<span class="setia-badge setia-published">✅ Published</span>'


def walkthrough(state: dict, position: int = 0) -> str:
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
        f'<div class="setia-sentence"{delay(position)}>{badge(state)} <b>{html.escape(state["source"])}</b>'
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


def delay(position: int) -> str:
    """Cards appear one after another; capped so long documents don't wait."""
    return f' style="animation-delay:{min(position, 12) * 60}ms"'


def check_card(state: dict, position: int = 0) -> str:
    """HTML for one checked pair: English and translation side by side, then any errors."""
    errors = [ErrorItem(**e) for e in state["history"][-1]["errors"]]
    if errors:
        badge_html = '<span class="setia-badge setia-review">Critical errors found</span>'
    else:
        badge_html = '<span class="setia-badge setia-published">No critical errors found</span>'
    found = "".join(
        f'<div class="setia-error" data-err="{number}"><span class="setia-chip" style="background:{CATEGORY_COLOURS[e.category]}">'
        f"{CATEGORY_LABELS[e.category]}</span>"
        + (f"Missing: “{html.escape(e.span)}”" if e.category == "removed_information" else html.escape(e.description))
        + "</div>"
        for number, e in enumerate(errors)
    )
    note = ""
    if state.get("unit") == "paragraph":
        note = '<div class="setia-note">Checked as one paragraph: the translation merged or split sentences.</div>'
    return (
        f'<div class="setia-sentence"{delay(position)}>{badge_html}<div class="setia-pair">'
        f'<div><div class="setia-label">English</div><div class="setia-text">{highlight(state["source"], errors, missing=True)}</div></div>'
        f'<div><div class="setia-label">Translation</div><div class="setia-text">{highlight(state["translation"], errors)}</div></div>'
        f"</div>{found}{note}</div>"
    )


def check_summary(result: dict) -> str:
    states = result["segments"]
    flagged = sum(bool(s["history"][-1]["errors"]) for s in states)
    # The numbers count up when shown (page script); the HTML already holds the final value
    count = lambda n: f'<span class="setia-count" data-to="{n}">{n}</span>'
    text = (
        f'<div class="setia-summary">{count(len(states))} sentences checked · {count(flagged)} with critical errors · '
        f"{count(len(states) - flagged)} with no critical errors found</div>"
    )
    english, translated = result["paragraphs"]
    if english != translated:
        text += (
            f'<div class="setia-warning">⚠️ The English has {english} paragraphs but the translation has '
            f"{translated}, so the whole text was checked as one block. This is a fallback, not the "
            "intended use: keep the same line breaks in both texts so each sentence is checked on its own.</div>"
        )
    return text


def segmented(labels: list[str], kind: str, target: str = "") -> str:
    """HTML for a pill segmented control; the page script wires it to Gradio (tabs or a radio)."""
    buttons = "".join(
        f'<button type="button"{" class=\'on\'" if i == 0 else ""}>{html.escape(label)}</button>' for i, label in enumerate(labels)
    )
    return f'<div class="setia-seg" data-kind="{kind}" data-target="{target}"><div class="setia-seg-pill"></div>{buttons}</div>'


# Shown the moment a check starts, until the results replace it
SKELETON = "".join(
    f'<div class="setia-sentence setia-skeleton"{delay(i)}><div class="setia-shimmer" style="width:22%"></div>'
    '<div class="setia-pair"><div><div class="setia-shimmer"></div><div class="setia-shimmer" style="width:70%"></div></div>'
    '<div><div class="setia-shimmer"></div><div class="setia-shimmer" style="width:60%"></div></div></div></div>'
    for i in range(3)
)
LANGUAGE_PILLS = ["中文", "Melayu", "தமிழ்"]


def build_app(run_document, check_document):
    """Builds the Gradio UI.

    Args:
        run_document: (text, lang) -> translate_document result, for the translate tab.
        check_document: (english, translation, lang) -> check_document result, for the check tab.
    """
    import gradio as gr

    def on_check(english: str, translation: str, language: str):
        # Nothing to check: leave the page exactly as it is
        if not english.strip() or not translation.strip():
            yield gr.update(), gr.update()
            return
        yield "", SKELETON
        result = check_document(english, translation, LANGUAGES[language])
        yield check_summary(result), "".join(check_card(s, i) for i, s in enumerate(result["segments"]))

    def on_translate(text: str, language: str):
        # Nothing to translate: leave the page exactly as it is
        if not text.strip():
            yield gr.update(), gr.update(), gr.update()
            return
        # The results section (title and colour legend) only exists once a translation is run
        title = '<div class="setia-results-title">What the workflow did:</div>' + legend()
        yield "", gr.update(), title + SKELETON
        result = run_document(text, LANGUAGES[language])
        states = result["segments"]
        yield summary(states), result["translation"], title + "".join(walkthrough(s, i) for i, s in enumerate(states))

    def language_control(elem_id: str):
        """Language pills, backed by a hidden radio that holds the value for Python."""
        gr.HTML(f'<div class="setia-field-label">Translation language</div>{segmented(LANGUAGE_PILLS, "radio", elem_id)}')
        return gr.Radio(list(LANGUAGES), value=list(LANGUAGES)[0], elem_id=elem_id, elem_classes=["setia-hidden"])

    with gr.Blocks(title=APP_NAME) as app:
        gr.HTML(
            f'<div class="setia-bar"></div><div class="setia-hero"><h1 class="setia-title">{APP_NAME}</h1>'
            f'<p class="setia-tagline">{SUBTITLE}</p></div>'
        )
        gr.HTML(f'<div class="setia-tabs">{segmented(["Check your translation", "Translate and check"], "views")}</div>')
        with gr.Column(elem_id="setia-view-0"):
            gr.HTML('<div class="setia-subtitle">Check your machine translation for critical errors</div>')
            with gr.Row(equal_height=True, elem_classes=["setia-panes"]):
                check_source = gr.Textbox(
                    label="English text", lines=12, max_lines=12, max_length=MAX_CHARACTERS, buttons=["copy"],
                    placeholder="Paste the original English text here", elem_classes=["setia-input"],
                )
                check_translation = gr.Textbox(
                    label="Translation", lines=12, max_lines=12, max_length=MAX_CHARACTERS, buttons=["copy"],
                    placeholder="Paste the translation to check here", elem_classes=["setia-input"],
                )
            with gr.Row(elem_classes=["setia-controls"]):
                with gr.Column(scale=2):
                    check_language = language_control("lang-check")
                check_button = gr.Button("Check Translation", variant="primary", scale=1)
                clear_button = gr.Button("Clear", scale=1)
                example_button = gr.Button("Try an example", scale=1)
            check_overview = gr.HTML()
            gr.HTML(legend())
            check_details = gr.HTML()
            check_button.click(
                on_check, [check_source, check_translation, check_language], [check_overview, check_details],
                show_progress="hidden", api_visibility="private",
            )
            clear_button.click(
                lambda: ("", "", "", ""), None, [check_source, check_translation, check_overview, check_details],
                api_visibility="private",
            )
            example_button.click(
                lambda language: (EXAMPLE_ENGLISH, EXAMPLE_TRANSLATIONS[LANGUAGES[language]]),
                check_language, [check_source, check_translation], api_visibility="private",
            )
        with gr.Column(elem_id="setia-view-1"):
            gr.HTML(
                '<div class="setia-subtitle setia-subtitle-wide">Use our model to translate your text to any of '
                "Singapore's official languages, and watch our LangGraph workflow<br>check every sentence for "
                "critical errors, retry with feedback when it finds one, and flag anything it can't fix for a "
                "person to review</div>"
            )
            # Same size side by side (like a translation tool); controls below both
            with gr.Row(equal_height=True, elem_classes=["setia-panes"]):
                source = gr.Textbox(
                    label="English text", lines=12, max_lines=12, max_length=MAX_CHARACTERS, buttons=["copy"],
                    placeholder="Paste the English text to translate here", elem_classes=["setia-input"],
                )
                # Editable, so an officer can correct sentences sent for human review
                translation = gr.Textbox(
                    label="Translation (you can correct it here)", lines=12, max_lines=12, buttons=["copy"],
                    placeholder="The translation will appear here once you input your English text",
                    elem_classes=["setia-input", "setia-no-count"],
                )
            with gr.Row(elem_classes=["setia-controls"]):
                with gr.Column(scale=2):
                    language = language_control("lang-translate")
                button = gr.Button("Translate and check", variant="primary", scale=1)
                translate_clear = gr.Button("Clear", scale=1)
            overview = gr.Markdown()
            details = gr.HTML()
            # Page-only events: no public API endpoints
            button.click(
                on_translate, [source, language], [overview, translation, details],
                show_progress="hidden", api_visibility="private",
            )
            translate_clear.click(
                lambda: ("", "", "", ""), None, [source, translation, overview, details], api_visibility="private"
            )
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


def preview_check(english: str, translation: str, lang: str) -> dict:
    """Simulates the check tab without models: a pair is flagged if its numbers differ."""
    from safetranslate.workflow.graph import pair_document

    pairs, counts = pair_document(english, translation)
    states = []
    for pair in pairs:
        english_numbers = set(re.findall(r"\d+", pair["source"]))
        wrong = [n for n in re.findall(r"\d+", pair["translation"]) if n not in english_numbers]
        errors = [ErrorItem(category="wrong_quantity", span=n, description=f"{n} does not appear in the English (simulated check)").model_dump() for n in wrong[:1]]
        states.append(pair | {"status": "checked", "history": [{"translation": pair["translation"], "errors": errors}]})
    return {"segments": states, "paragraphs": counts}


def main() -> None:
    import gradio as gr

    if "--preview" in sys.argv:
        build_app(preview_document, preview_check).launch(theme=theme(), css=CSS, head=HEAD)
        return

    from safetranslate.alter.generate import make_client
    from safetranslate.config import load_harness_config, load_workflow_config
    from safetranslate.workflow.graph import build_graph, check_document, translate_document

    config = load_workflow_config(sys.argv[1])
    harness = load_harness_config(config.harness_config)
    translator, evaluator = harness.systems[config.translator], harness.systems[config.evaluator]
    graph = build_graph(
        (make_client(translator), translator), (make_client(evaluator), evaluator), config.max_attempts
    )
    app = build_app(
        lambda text, lang: translate_document(graph, text, lang),
        lambda english, translation, lang: check_document(graph, english, translation, lang),
    )
    app.launch(theme=theme(), css=CSS, head=HEAD)


if __name__ == "__main__":
    main()
