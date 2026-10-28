"""Command-line front end for the moderation queue.

    python cli.py queue                       # listings that need a human (review / block)
    python cli.py show L-003                  # every check for one listing
    python cli.py annotate L-003 out.jpg      # photo with the model's boxes drawn on it
    python cli.py quality                     # mAP@0.5 of model boxes vs human ground truth
    python cli.py videos                      # worst frame verdict per listing video
    python cli.py spoken                      # voice-over chunks asking to move off-platform
    MARKET_CATALOG=pxt://<org>:<db> python cli.py queue
"""

from __future__ import annotations

import argparse
import json
import os

import pixeltable as pxt
from pixeltable.functions import vision

from app import Listings, TableModel, VideoFrames, VoiceClips
from udfs import LABELS, worst_decision

CATALOG = os.environ.get('MARKET_CATALOG', 'marketplace')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('queue')
    s = sub.add_parser('show')
    s.add_argument('listing_id')
    a = sub.add_parser('annotate')
    a.add_argument('listing_id')
    a.add_argument('out')
    sub.add_parser('quality')
    sub.add_parser('videos')
    sub.add_parser('spoken')
    args = ap.parse_args()
    TableModel.bind_all(CATALOG)

    if args.cmd == 'queue':
        q = Listings.where(Listings.status != 'allow').order_by(Listings.listing_id)
        for r in q.select(Listings.listing_id, Listings.seller, Listings.title, Listings.status, Listings.decision.reasons).collect():
            print(f"{r['listing_id']}  {r['status']:6s}  {r['seller']:5s} {r['title']}\n          {', '.join(r['decision_reasons'])}")
    elif args.cmd == 'show':
        q = Listings.where(Listings.listing_id == args.listing_id)
        for r in q.select(Listings.listing_id, Listings.title, Listings.category, Listings.status, Listings.decision, Listings.text_contact, Listings.verdict).collect():
            print(json.dumps(r, indent=2, default=str))
    elif args.cmd == 'annotate':
        rows = Listings.where(Listings.listing_id == args.listing_id).select(Listings.annotated).collect()
        rows[0]['annotated'].save(args.out)
        print(f'wrote {args.out}')
    elif args.cmd == 'quality':
        for r in Listings.where(Listings.gt != None).select(map50=vision.mean_ap(Listings.detections)).collect():  # noqa: E711
            print({LABELS[int(k)]: round(v, 3) for k, v in r['map50'].items()})
    elif args.cmd == 'videos':
        q = VideoFrames.group_by(VideoFrames.listing_id)
        for r in q.select(VideoFrames.listing_id, worst=worst_decision(VideoFrames.frame_decision), frames=pxt.functions.count(VideoFrames.pos)).collect():
            print(f"{r['listing_id']}  worst frame: {r['worst']}  ({r['frames']} frames)")
    elif args.cmd == 'spoken':
        q = VoiceClips.order_by(VoiceClips.listing_id, VoiceClips.segment_start)
        for r in q.select(VoiceClips.listing_id, VoiceClips.segment_start, VoiceClips.said, VoiceClips.contact).collect():
            hits = r['contact']['phones'] + r['contact']['off_platform']
            if hits:
                print(f"{r['listing_id']} @ {r['segment_start']:.0f}s  {r['said']}\n          -> {hits}")


if __name__ == '__main__':
    main()
