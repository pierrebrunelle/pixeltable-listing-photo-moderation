"""UDFs for the listing moderation pipeline (no tables here; see app.py)."""

from __future__ import annotations

import json
import re

import pixeltable as pxt

# Fixed label vocabulary shared by the vision prompt, the ground truth and eval_detections.
LABELS = ['person', 'animal', 'vehicle', 'contact_info']
LABEL_IDS = {name: i for i, name in enumerate(LABELS)}
HARD_FLAGS = {'weapon', 'drugs', 'adult', 'gore'}

_PHONE = re.compile(r'(\+?\d[\d\s().-]{6,}\d)')
_SPOKEN_DIGITS = re.compile(
    r'\b(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)(?:[\s,-]+(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)){5,}\b',
    re.IGNORECASE,
)
_OFF_PLATFORM = re.compile(r'\b(whatsapp|telegram|signal|venmo|zelle|cash ?app|gift ?card|text me|dm me|e-?mail me)\b', re.IGNORECASE)


@pxt.udf
def parse_verdict(raw: str) -> pxt.Json:
    """Parse the vision model's JSON answer; tolerate a fenced code block."""
    raw = raw.strip()
    if raw.startswith('```'):
        raw = raw.strip('`').removeprefix('json').strip()
    return json.loads(raw)


@pxt.udf
def object_boxes(verdict: pxt.Json) -> pxt.Json:
    """Relative [x1, y1, x2, y2] boxes of the objects the model reported (same order as object_labels)."""
    return [[float(c) for c in o['box']] for o in verdict.get('objects', [])]


@pxt.udf
def object_labels(verdict: pxt.Json) -> pxt.Json:
    return [LABEL_IDS.get(o['label'], -1) for o in verdict.get('objects', [])]


@pxt.udf
def object_scores(verdict: pxt.Json) -> pxt.Json:
    return [float(o.get('confidence', 0.5)) for o in verdict.get('objects', [])]


@pxt.udf
def to_pixels(boxes: pxt.Json, width: int, height: int) -> pxt.Json:
    """Relative xyxy boxes -> integer pixel xyxy boxes. A clipped-away box (None) becomes [0, 0, 0, 0]
    so the list stays aligned with its labels."""
    out = []
    for b in boxes or []:
        if b is None:
            out.append([0, 0, 0, 0])
        else:
            out.append([round(b[0] * width), round(b[1] * height), round(b[2] * width), round(b[3] * height)])
    return out


@pxt.udf
def label_names(ids: pxt.Json) -> pxt.Json:
    return [LABELS[i] if 0 <= i < len(LABELS) else 'other' for i in ids or []]


@pxt.udf
def contact_info_in_text(text: str) -> pxt.Json:
    """Phone numbers (written or spoken) and off-platform payment/contact words in free text."""
    return {
        'phones': _PHONE.findall(text or '') + _SPOKEN_DIGITS.findall(text or ''),
        'off_platform': sorted({m.lower() for m in _OFF_PLATFORM.findall(text or '')}),
    }


@pxt.udf
def flagged_categories(moderation: pxt.Json) -> pxt.Json:
    """Names of the OpenAI moderation categories that fired."""
    res = moderation['results'][0]
    # The API reports some categories twice ('harassment/threatening' and 'harassment_threatening').
    return sorted({k.replace('_', '/') for k, v in res['categories'].items() if v})


@pxt.udf
def decide(category: str, text_flags: pxt.Json, verdict: pxt.Json, text_contact: pxt.Json) -> pxt.Json:
    """Combine text moderation, the vision verdict and contact-info checks into allow / review / block."""
    reasons: list[str] = []
    if text_flags:
        reasons.append('text:' + ','.join(text_flags))
    vision = verdict.get('verdict', 'review')
    reasons += [f'vision:{c}' for c in verdict.get('policy_flags', [])]
    if text_contact['phones'] or text_contact['off_platform']:
        reasons.append('text:off_platform_contact')
    labels = {o['label'] for o in verdict.get('objects', [])}
    if 'contact_info' in labels:
        reasons.append('photo:contact_info')
    if verdict.get('matches_category') is False:
        reasons.append(f'photo:not_a_{category}')
    # Only hard policy violations block on the photo alone; people, animals and wrong items go to review.
    hard = HARD_FLAGS.intersection(verdict.get('policy_flags', []))
    blocking = any(r.startswith('text:') and 'off_platform' not in r for r in reasons) or bool(hard)
    if blocking or 'photo:contact_info' in reasons and 'text:off_platform_contact' in reasons:
        decision = 'block'
    elif reasons or vision == 'review':
        decision = 'review'
    else:
        decision = 'allow'
    return {'decision': decision, 'reasons': reasons}


@pxt.udf
def frame_flags(raw: str) -> pxt.Json:
    raw = raw.strip()
    if raw.startswith('```'):
        raw = raw.strip('`').removeprefix('json').strip()
    return json.loads(raw)


@pxt.uda
class worst_decision(pxt.Aggregator):
    """Aggregate per-frame / per-clip decisions: block > review > allow."""

    RANK = {'allow': 0, 'review': 1, 'block': 2}

    def __init__(self) -> None:
        self.best = 'allow'

    def update(self, decision: str | None) -> None:
        if decision is not None and self.RANK.get(decision, 1) > self.RANK[self.best]:
            self.best = decision

    def value(self) -> str:
        return self.best


@pxt.udf
def listing_text(title: str, description: str) -> str:
    return f'{title}\n\n{description}'


@pxt.udf
def vision_prompt(title: str, category: str) -> str:
    return (
        f'Marketplace listing in category "{category}" titled "{title}". Review the photo for our policy: '
        'block weapons, drugs, adult content, gore; review photos that show phone numbers, e-mail, QR codes or '
        'messenger handles (off-platform contact), live animals for sale, identifiable people, or that do not '
        'show the listed item. For every person, animal, vehicle and contact_info text overlay, return a box as '
        'relative [x1, y1, x2, y2] in 0..1.'
    )
