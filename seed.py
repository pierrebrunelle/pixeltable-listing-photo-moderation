"""Load nine demo listings (photos) and two listing videos with seller voice-overs.

Photos are COCO images from the Pixeltable repo plus one edited copy with a phone-number banner;
one listing references its photo by URL. Human ground-truth boxes (people, animals, vehicles,
contact-info overlays) come from data/ground_truth.json.
Cost: 9 moderation + 9 gpt-4o-mini vision calls for photos; per video ~6 frame checks plus
~2 transcriptions and ~2 moderations.

    python seed.py            # local catalog 'marketplace'
    python seed.py pxt://<your-org>:<your-db>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app import Listings, ListingVideos, TableModel

TARGET = sys.argv[1] if len(sys.argv) > 1 else 'marketplace'
HERE = Path(__file__).parent
PHOTOS = HERE / 'data' / 'photos'
GT = json.loads((HERE / 'data' / 'ground_truth.json').read_text())
ZEBRA_URL = 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/images/000000000034.jpg'

LISTINGS = [
    ('L-001', 'ana', 'Set of three ceramic bowls', 'Glazed bowls, dishwasher safe. Pickup or shipping.', 'kitchen', 24.0, 'coco-000000000009.jpg', '000000000009'),
    ('L-002', 'ben', 'Handmade vase with succulent', 'Small stoneware vase, plant included. No chips.', 'home-decor', 20.0, 'coco-000000000030.jpg', '000000000030'),
    ('L-003', 'ben', 'Handmade vase - cheaper off the app', 'Same vase, 15 if you text me on WhatsApp 555-0142 and pay with Zelle.', 'home-decor', 15.0, 'vase-with-phone-banner.jpg', 'vase-with-phone-banner'),
    ('L-004', 'cara', 'Large golf umbrella', 'Wind-proof 62 inch umbrella, used twice.', 'outdoor', 18.0, 'coco-000000000036.jpg', '000000000036'),
    ('L-005', 'dev', 'Puppy for sale, 8 weeks', 'Friendly pup, vaccinated. Serious buyers only.', 'pets', 300.0, 'coco-000000000042.jpg', '000000000042'),
    ('L-006', 'eli', 'Antique street clock', 'Cast iron post clock, works, local pickup.', 'collectibles', 450.0, 'coco-000000000064.jpg', '000000000064'),
    ('L-007', 'fay', 'Sewing thread bundle, 40 spools', 'Assorted colours, mostly full spools.', 'crafts', 12.0, 'sewing-threads.jpg', None),
    ('L-008', 'gus', 'Serving bowls, firm price', 'Firm price. Lowball me again and I will find where you live and hurt you.', 'kitchen', 30.0, 'coco-000000000009.jpg', '000000000009'),
    ('L-009', 'hal', 'Zebra print rug', 'Faux zebra rug, 5x7 ft.', 'home-decor', 80.0, ZEBRA_URL, '000000000034'),
]

VIDEOS = [
    ('L-007', 'fay', 'listing-threads.mp4'),
    ('L-002', 'ben', 'listing-vase.mp4'),
]


def listing_rows() -> list[dict]:
    rows = []
    for lid, seller, title, desc, cat, price, photo, gt_key in LISTINGS:
        rows.append({
            'listing_id': lid, 'seller': seller, 'title': title, 'description': desc, 'category': cat,
            'price': price, 'photo': photo if photo.startswith('http') else str(PHOTOS / photo),
            'gt': GT.get(gt_key) if gt_key else None,
        })
    return rows


def video_rows() -> list[dict]:
    return [{'listing_id': lid, 'seller': s, 'video': str(HERE / 'data' / 'videos' / f)} for lid, s, f in VIDEOS]


if __name__ == '__main__':
    TableModel.bind_all(TARGET)
    print(Listings.insert(listing_rows(), on_error='ignore'))
    print(ListingVideos.insert(video_rows(), on_error='ignore'))
