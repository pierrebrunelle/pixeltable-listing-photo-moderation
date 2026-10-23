"""Marketplace listing photo and video moderation built with Pixeltable.

Sellers upload a photo (and optionally a short video with a voice-over) for each listing.
Computed columns run the checks a trust-and-safety queue needs:

- text moderation of title + description (`openai.moderations`, omni-moderation-latest)
- a vision verdict on the photo (`gpt-4o-mini`, strict JSON schema): allow/review/block,
  policy flags, whether the photo matches the category, and boxes for people, animals,
  vehicles and contact-info overlays
- box post-processing with `pixeltable.functions.vision` (clip to canvas, draw, evaluate
  against human ground truth with `eval_detections` / `mean_ap`)
- videos are sampled every 2 s (`frame_iterator`) and checked frame by frame; their audio
  track is extracted, split (`audio_splitter`), transcribed and scanned for phone numbers and
  off-platform payment requests

Layout: UDFs/UDAs live in udfs.py; this file holds table models and @pxt.query functions.

    pxt schema update app.py marketplace
    python seed.py
    python cli.py queue
"""

from __future__ import annotations

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.functions import openai, vision

from udfs import (
    LABELS,
    contact_info_in_text,
    decide,
    flagged_categories,
    frame_flags,
    label_names,
    listing_text,
    object_boxes,
    object_labels,
    object_scores,
    parse_verdict,
    to_pixels,
    vision_prompt,
    worst_decision,
)

VISION_MODEL = 'gpt-4o-mini'
MODERATION_MODEL = 'omni-moderation-latest'
TRANSCRIBE_MODEL = 'gpt-4o-mini-transcribe'

VERDICT_SCHEMA = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'photo_verdict',
        'strict': True,
        'schema': {
            'type': 'object',
            'additionalProperties': False,
            'required': ['verdict', 'matches_category', 'policy_flags', 'objects', 'summary'],
            'properties': {
                'verdict': {'type': 'string', 'enum': ['allow', 'review', 'block']},
                'matches_category': {'type': 'boolean'},
                'policy_flags': {
                    'type': 'array',
                    'items': {
                        'type': 'string',
                        'enum': [
                            'off_platform_contact', 'live_animal', 'identifiable_person', 'wrong_item',
                            'weapon', 'drugs', 'adult', 'gore',
                        ],
                    },
                },
                'objects': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'additionalProperties': False,
                        'required': ['label', 'box', 'confidence'],
                        'properties': {
                            'label': {'type': 'string', 'enum': LABELS},
                            'box': {'type': 'array', 'items': {'type': 'number'}},
                            'confidence': {'type': 'number'},
                        },
                    },
                },
                'summary': {'type': 'string'},
            },
        },
    },
}

FRAME_PROMPT = (
    'This is one frame of a marketplace listing video. Answer in JSON with keys "decision" '
    '("allow", "review" or "block"), "contact_info" (true if a phone number, e-mail, QR code or messenger '
    'handle is visible) and "what" (max 8 words).'
)

TableModel = pxt.model_base()


class Listings(TableModel, name='listings'):
    """One marketplace listing: text fields plus one photo, with optional human-drawn boxes."""

    listing_id = pxt.Column(type=pxt.String, primary_key=True)
    seller: pxt.String
    title: pxt.String
    description: pxt.String
    category: pxt.String
    price: pxt.Float
    photo: pxt.Image
    gt: pxt.Json | None  # {'labels': [int], 'boxes': [[x, y, w, h] relative]} from a human reviewer

    photo_meta = pxtf.image.get_metadata(photo)
    small = pxtf.image.thumbnail(photo, size=(512, 512))
    text = listing_text(title, description)
    text_mod = openai.moderations(text, model=MODERATION_MODEL)
    text_flags = flagged_categories(text_mod)
    text_contact = contact_info_in_text(text)
    vision_raw = openai.chat_completions(
        [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': vision_prompt(title, category)},
                    {'type': 'image_url', 'image_url': small},
                ],
            }
        ],
        model=VISION_MODEL,
        model_kwargs={'response_format': VERDICT_SCHEMA, 'max_tokens': 500, 'temperature': 0},
    )
    verdict = parse_verdict(vision_raw.choices[0].message.content.astype(pxt.String))
    pred_labels = object_labels(verdict)
    pred_scores = object_scores(verdict)
    pred_rel = vision.bboxes_clip_to_canvas(object_boxes(verdict), format='xyxy')
    pred_px = to_pixels(pred_rel, photo.width, photo.height)
    gt_px = to_pixels(vision.bboxes_convert(gt.boxes, src_format='xywh', dst_format='xyxy'), photo.width, photo.height)
    detections = vision.eval_detections(
        pred_bboxes=pred_px, pred_labels=pred_labels, pred_scores=pred_scores,
        gt_bboxes=gt_px, gt_labels=gt.labels, min_iou=0.5,
    )
    annotated = vision.bboxes_draw(photo, pred_px, labels=label_names(pred_labels), width=3)
    decision = decide(category, text_flags, verdict, text_contact)
    status = decision.decision.astype(pxt.String)

    __indexes__ = [pxt.BtreeIndex(status), pxt.BtreeIndex(seller)]


class ListingVideos(TableModel, name='listing_videos'):
    """Optional short video per listing (walk-around + seller voice-over)."""

    listing_id = pxt.Column(type=pxt.String, primary_key=True)
    seller: pxt.String
    video: pxt.Video

    duration = pxtf.video.get_duration(video)
    audio = pxtf.video.extract_audio(video, format='mp3')


class VideoFrames(
    TableModel,
    name='video_frames',
    base=ListingVideos,
    iterator=pxtf.video.frame_iterator(ListingVideos.video, fps=0.5),
):
    """Component view: one frame every 2 s, checked by the vision model."""

    t = frame_attrs.time.astype(pxt.Float)  # type: ignore[name-defined]  # noqa: F821
    small = frame.resize((320, 214))  # type: ignore[name-defined]  # noqa: F821
    check_raw = openai.chat_completions(
        [{'role': 'user', 'content': [{'type': 'text', 'text': FRAME_PROMPT}, {'type': 'image_url', 'image_url': small}]}],
        model=VISION_MODEL,
        model_kwargs={'response_format': {'type': 'json_object'}, 'max_tokens': 80, 'temperature': 0},
    )
    check = frame_flags(check_raw.choices[0].message.content.astype(pxt.String))
    frame_decision = check.decision.astype(pxt.String)


class VoiceClips(
    TableModel,
    name='voice_clips',
    base=ListingVideos,
    iterator=pxtf.audio.audio_splitter(ListingVideos.audio, duration=6.0, overlap=0.0, min_segment_duration=1.0),
):
    """Component view: ~6 s chunks of the voice-over, transcribed and scanned for off-platform contact."""

    heard = openai.transcriptions(audio_segment, model=TRANSCRIBE_MODEL)  # type: ignore[name-defined]  # noqa: F821
    said = heard.text.astype(pxt.String)
    contact = contact_info_in_text(said)
    said_mod = openai.moderations(said, model=MODERATION_MODEL)


@pxt.query
def review_queue():
    """Listings a human must look at (review or block), newest reasons first."""
    return (
        Listings.where(Listings.status != 'allow')
        .order_by(Listings.listing_id)
        .select(Listings.listing_id, Listings.seller, Listings.title, Listings.status, Listings.decision.reasons)
    )


@pxt.query
def listing_report(listing_id: str):
    """Everything the checks found for one listing."""
    return Listings.where(Listings.listing_id == listing_id).select(
        Listings.listing_id, Listings.title, Listings.category, Listings.status,
        Listings.decision, Listings.text_flags, Listings.text_contact, Listings.verdict,
    )


@pxt.query
def seller_listings(seller: str):
    """All listings of one seller with their status (seller index)."""
    return Listings.where(Listings.seller == seller).order_by(Listings.listing_id).select(
        Listings.listing_id, Listings.title, Listings.status
    )


@pxt.query
def detection_quality():
    """mAP@0.5 of the vision model's boxes against the human ground truth."""
    return Listings.where(Listings.gt != None).select(map50=vision.mean_ap(Listings.detections))  # noqa: E711


@pxt.query
def video_verdicts():
    """Worst frame decision per listing video (block > review > allow)."""
    return VideoFrames.group_by(VideoFrames.listing_id).select(
        VideoFrames.listing_id, worst=worst_decision(VideoFrames.frame_decision), frames=pxtf.count(VideoFrames.pos)
    )


@pxt.query
def spoken_contact():
    """Voice-over chunks where the seller asks to move off-platform."""
    return VoiceClips.order_by(VoiceClips.listing_id, VoiceClips.segment_start).select(
        VoiceClips.listing_id, VoiceClips.segment_start, VoiceClips.said, VoiceClips.contact
    )
