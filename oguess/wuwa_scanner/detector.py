"""
Wuthering Waves Inventory Item Detector  v3
============================================
Detects item slots, reads quantities (OCR), perceptual-hashes icons,
and matches against a local dataset of named items.

Dependencies: opencv-python-headless, Pillow, imagehash, pytesseract
              + system tesseract-ocr (apt install tesseract-ocr)
"""

import cv2
import numpy as np
import pytesseract
import statistics
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from PIL import Image
import imagehash

DATASET_PATH = Path(__file__).parent / "data" / "item_dataset.json"
CROPS_DIR    = Path(__file__).parent / "item_crops"


# ─────────────────────────────────────────────
#  Data types
# ─────────────────────────────────────────────

@dataclass
class ItemSlot:
    index:      int
    bbox:       tuple          # (x, y, w, h) in original image coords
    quantity:   int
    image_hash: str
    name:       Optional[str] = None
    crop_path:  Optional[str] = None
    is_new:     bool = False   # gold "新" (new item) badge


# ─────────────────────────────────────────────
#  Dataset persistence
# ─────────────────────────────────────────────

def load_dataset() -> dict:
    """Load item hash→name mapping from disk."""
    if DATASET_PATH.exists():
        with open(DATASET_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_dataset(db: dict) -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def register_item(image_hash: str, name: str, db: dict) -> dict:
    """Add or update an item name in the dataset."""
    if image_hash in db:
        db[image_hash]["seen"] += 1
        db[image_hash]["name"] = name
    else:
        db[image_hash] = {"name": name, "seen": 1}
    save_dataset(db)
    return db


# ─────────────────────────────────────────────
#  Image loading & preprocessing
# ─────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    return img


def preprocess(img: np.ndarray) -> np.ndarray:
    """Light bilateral filter to smooth noise while preserving edges."""
    return cv2.bilateralFilter(img, 5, 75, 75)


# ─────────────────────────────────────────────
#  Grid detection internals
# ─────────────────────────────────────────────

def _nms(boxes: np.ndarray, iou_thresh: float = 0.3) -> list:
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas  = (x2-x1)*(y2-y1)
    order  = areas.argsort()[::-1]   # largest first
    keep   = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return keep


def _cluster(values: list, min_gap: float) -> list:
    """Cluster 1-D positions; returns centroid of each cluster."""
    if not values:
        return []
    sv = sorted(values)
    clusters = [[sv[0]]]
    for v in sv[1:]:
        if v - clusters[-1][-1] < min_gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [int(np.mean(c)) for c in clusters]


def _fill_gaps(centers: list, med_span: int) -> list:
    """
    Insert inferred column centres where the gap between detected columns
    is ~2× the typical spacing (a column was missed by the detector).
    """
    if len(centers) < 2:
        return centers
    diffs   = [centers[i+1] - centers[i] for i in range(len(centers)-1)]
    typical = [d for d in diffs if d < med_span * 2.5]
    if not typical:
        return centers
    med_sp  = statistics.median(typical)
    filled  = list(centers)
    for i, d in enumerate(diffs):
        if d > med_sp * 1.6:
            n_missing = round(d / med_sp) - 1
            for k in range(1, n_missing + 1):
                filled.append(centers[i] + int(k * med_sp))
    return sorted(set(filled))


# ─────────────────────────────────────────────
#  Slot detector
# ─────────────────────────────────────────────

def detect_item_slots(img: np.ndarray, debug: bool = False) -> list:
    """
    Detect WuWa inventory item slot bounding boxes.

    Works on any screenshot resolution (tested on 1080p, 1440p, ultrawide).
    Returns list of dicts {x, y, w, h} sorted in row-major reading order.
    """
    h, w = img.shape[:2]

    # ── Crop to inventory grid region (skip sidebars & info panel) ──────────
    left   = int(w * 0.06)
    right  = int(w * 0.76)
    top    = int(h * 0.04)
    bottom = int(h * 0.96)
    roi    = img[top:bottom, left:right]

    # ── Edge detection ────────────────────────────────────────────────────────
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges   = cv2.Canny(blurred, 25, 90)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges   = cv2.dilate(edges, kernel, iterations=1)

    # ── Find rectangular card-like contours ──────────────────────────────────
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rh, rw      = roi.shape[:2]
    min_a, max_a = rh * rw * 0.002, rh * rw * 0.10

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_a < area < max_a):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if not (0.65 < bw / bh < 1.45):   # roughly square
            continue
        candidates.append((x, y, bw, bh))

    if not candidates:
        return []

    # ── Determine modal card size via median ─────────────────────────────────
    med_w = int(np.median([c[2] for c in candidates]))
    med_h = int(np.median([c[3] for c in candidates]))

    tol = 0.35
    filtered = [c for c in candidates
                if abs(c[2]-med_w)/med_w < tol and abs(c[3]-med_h)/med_h < tol]
    if not filtered:
        return []

    # ── Non-maximum suppression ───────────────────────────────────────────────
    boxes = np.array([[x, y, x+bw, y+bh] for x, y, bw, bh in filtered])
    keep  = _nms(boxes)
    filtered = [filtered[i] for i in keep]

    # ── Build regular grid from clustered positions ───────────────────────────
    col_ctr = _cluster([c[0] for c in filtered], med_w * 0.5)
    row_ctr = _cluster([c[1] for c in filtered], med_h * 0.5)
    col_ctr = _fill_gaps(col_ctr, med_w)   # infer any missed columns

    # ── Translate ROI coordinates → full-image coordinates ───────────────────
    result, seen = [], set()
    for ry in row_ctr:
        for cx in col_ctr:
            if (cx, ry) in seen:
                continue
            seen.add((cx, ry))
            result.append({"x": cx + left, "y": ry + top, "w": med_w, "h": med_h})

    if debug:
        dbg = roi.copy()
        for r in result:
            rx, ry2 = r["x"]-left, r["y"]-top
            cv2.rectangle(dbg, (rx, ry2), (rx+med_w, ry2+med_h), (0, 255, 0), 2)
        cv2.imwrite("/tmp/wuwa_debug_slots.png", dbg)

    return result   # already row-major


# ─────────────────────────────────────────────
#  Quantity OCR
# ─────────────────────────────────────────────

def _safe_ocr(arr: np.ndarray, cfg: str) -> str:
    """Tesseract wrapper that returns '' on empty image or error."""
    if arr is None or arr.sum() == 0:
        return ""
    try:
        return pytesseract.image_to_string(arr, config=cfg).strip()
    except Exception:
        return ""


def _scan_qty_hits(img: np.ndarray, slot: dict, threshold: int) -> tuple:
    """
    Scan horizontal strips in the quantity-text zone of the card.
    Returns (hits_nearest, hits_cubic) where each is a list of (offset, digits, white_count).
    """
    x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
    cfg         = "--psm 7 -c tessedit_char_whitelist=0123456789"
    hits_nn, hits_cub = [], []

    # Quantity text lives at ~72–94 % of the detected card height
    for off in range(int(h * 0.72), int(h * 0.94), 4):
        strip = img[y+off : y+off+28, x : x+w]
        if strip.shape[0] < 4 or strip.shape[1] < 4:
            continue
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        wc   = int((gray > threshold).sum())
        if wc < 6:
            continue
        _, white = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # NEAREST: avoids digit-blending artefacts (preferred)
        up = cv2.resize(white, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        d  = "".join(c for c in _safe_ocr(up, cfg) if c.isdigit())
        if d:
            hits_nn.append((off, d, wc))

        # CUBIC: smoother, better for dim / faint digits
        up = cv2.resize(white, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        d  = "".join(c for c in _safe_ocr(up, cfg) if c.isdigit())
        if d:
            hits_cub.append((off, d, wc))

    return hits_nn, hits_cub


def _best_from_hits(hits: list) -> int:
    """
    From a list of (offset, digits, white_count) hits:
    1. Group by consecutive offsets (gap ≤ 14 px).
    2. Pick the group with the highest max white-count.
    3. Within that group prefer the longest digit string ≤ 4 digits,
       breaking ties by white-count.
    """
    if not hits:
        return 0
    groups = [[hits[0]]]
    for item in hits[1:]:
        if item[0] - groups[-1][-1][0] <= 14:
            groups[-1].append(item)
        else:
            groups.append([item])
    scores   = [max(g, key=lambda t: t[2])[2] for g in groups]
    best_grp = groups[scores.index(max(scores))]
    best     = max(best_grp, key=lambda t: (len(t[1]) if len(t[1]) <= 4 else 0, t[2]))
    return int(best[1])


def read_quantity(img: np.ndarray, slot: dict) -> int:
    """
    Read the quantity shown at the bottom of a WuWa item card.

    Strategy:
    - Run Tesseract on upscaled strips around the qty zone.
    - Try two brightness thresholds (for bright and dim text).
    - Use NEAREST interpolation as primary, CUBIC as secondary.
    - Prefer NEAREST result when both disagree and NEAREST is shorter
      (CUBIC sometimes concatenates adjacent glyphs into extra digits).
    """
    for threshold in (185, 155):
        nn, cub = _scan_qty_hits(img, slot, threshold)
        rn = _best_from_hits(nn)
        rc = _best_from_hits(cub)

        if rn == 0 and rc == 0:
            continue   # try lower threshold

        if rn == 0:   return rc
        if rc == 0:   return rn
        if rn == rc:  return rn

        # Disagreement: prefer NEAREST when result is shorter or equal length
        return rn if len(str(rn)) <= len(str(rc)) else rc

    return 0


# ─────────────────────────────────────────────
#  Perceptual hashing
# ─────────────────────────────────────────────

def hash_slot(slot_crop: np.ndarray, hash_size: int = 16) -> str:
    """
    Compute a perceptual hash of the item icon region
    (inner 85 % width, top 78 % height — excludes qty strip and badge).
    """
    h, w = slot_crop.shape[:2]
    icon = slot_crop[int(h*0.05) : int(h*0.78),
                     int(w*0.05) : int(w*0.92)]
    pil  = Image.fromarray(cv2.cvtColor(icon, cv2.COLOR_BGR2RGB))
    return str(imagehash.phash(pil, hash_size=hash_size))


def find_best_match(query_hash: str, db: dict, threshold: int = 12) -> Optional[str]:
    """Return the dataset key whose perceptual hash is closest to query_hash."""
    if not db:
        return None
    query    = imagehash.hex_to_hash(query_hash)
    best_d   = threshold + 1
    best_key = None
    for key in db:
        try:
            dist = query - imagehash.hex_to_hash(key)
            if dist < best_d:
                best_d, best_key = dist, key
        except Exception:
            continue
    return best_key


# ─────────────────────────────────────────────
#  "New item" badge detection
# ─────────────────────────────────────────────

def detect_new_badge(slot_crop: np.ndarray) -> bool:
    """Detect the gold '新' badge in the top-left corner of a card."""
    h, w   = slot_crop.shape[:2]
    corner = slot_crop[:int(h*0.25), :int(w*0.35)]
    hsv    = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
    mask   = cv2.inRange(hsv, np.array([15, 100, 150]), np.array([35, 255, 255]))
    return int(mask.sum()) > 200 * 255


# ─────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────

def scan_screenshot(image_path: str, debug: bool = False) -> list:
    """
    Full pipeline:
        load image → preprocess → detect slots → OCR qty
        → perceptual hash → match against dataset → return results.

    Args:
        image_path: Path to a WuWa inventory screenshot (any resolution).
        debug:      If True, saves /tmp/wuwa_debug_slots.png with slot overlays.

    Returns:
        List of ItemSlot objects in row-major (left→right, top→bottom) order.
        Unknown items have name=None and a crop saved to item_crops/.
    """
    db  = load_dataset()
    img = load_image(image_path)
    img = preprocess(img)

    slot_boxes = detect_item_slots(img, debug=debug)
    results    = []

    for idx, box in enumerate(slot_boxes):
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        ih, iw     = img.shape[:2]
        x1 = max(0, x);    y1 = max(0, y)
        x2 = min(iw, x+w); y2 = min(ih, y+h)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        phash  = hash_slot(crop)
        qty    = read_quantity(img, box)
        is_new = detect_new_badge(crop)

        matched = find_best_match(phash, db)
        name    = db[matched]["name"] if matched else None

        crop_path = None
        if name is None:
            CROPS_DIR.mkdir(parents=True, exist_ok=True)
            crop_path = str(CROPS_DIR / f"unknown_{phash[:12]}.png")
            cv2.imwrite(crop_path, crop)

        results.append(ItemSlot(
            index=idx, bbox=(x1, y1, w, h),
            quantity=qty, image_hash=phash,
            name=name, crop_path=crop_path, is_new=is_new,
        ))

    return results
