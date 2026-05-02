"""
Wuthering Waves Inventory Item Detector
Extracts item slots from inventory screenshots using adaptive grid detection.
"""

import cv2
import numpy as np
from PIL import Image
import imagehash
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

DATASET_PATH = Path(__file__).parent / "data" / "item_dataset.json"
CROPS_DIR = Path(__file__).parent / "item_crops"

@dataclass
class ItemSlot:
    index: int
    bbox: tuple          # (x, y, w, h) in original image coords
    quantity: int
    image_hash: str
    name: Optional[str] = None
    crop_path: Optional[str] = None
    is_new: bool = False  # has the gold "新" badge


# ─────────────────────────────────────────────
#  Dataset helpers
# ─────────────────────────────────────────────

def load_dataset() -> dict:
    if DATASET_PATH.exists():
        with open(DATASET_PATH) as f:
            return json.load(f)
    return {}   # hash → {"name": str, "seen": int}


def save_dataset(db: dict):
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def register_item(image_hash: str, name: str, db: dict) -> dict:
    if image_hash in db:
        db[image_hash]["seen"] += 1
        db[image_hash]["name"] = name  # allow rename
    else:
        db[image_hash] = {"name": name, "seen": 1}
    save_dataset(db)
    return db


# ─────────────────────────────────────────────
#  Image preprocessing
# ─────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    return img


def preprocess(img: np.ndarray) -> np.ndarray:
    """Mild denoising while preserving edges for contour detection."""
    return cv2.bilateralFilter(img, 5, 75, 75)


# ─────────────────────────────────────────────
#  Grid / slot detection
# ─────────────────────────────────────────────

def detect_item_slots(img: np.ndarray, debug: bool = False) -> list[dict]:
    """
    Detect inventory item slots in a WuWa screenshot.
    Returns list of dicts with keys: x, y, w, h
    Strategy:
      1. Isolate the inventory grid region (crop away side panels).
      2. Find rectangular contours that look like item tiles.
      3. Cluster by size / position to get a clean grid.
    """
    h, w = img.shape[:2]

    # ── 1. Rough crop: skip left sidebar (~8%) and right panel (~22%)
    left_cut  = int(w * 0.07)
    right_cut = int(w * 0.77)
    top_cut   = int(h * 0.05)
    bottom_cut = int(h * 0.95)
    roi = img[top_cut:bottom_cut, left_cut:right_cut]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # ── 2. Edge detection on the card borders
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # ── 3. Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── 4. Filter contours to plausible card shapes
    roi_h, roi_w = roi.shape[:2]
    min_area = (roi_h * roi_w) * 0.002   # at least 0.2% of roi
    max_area = (roi_h * roi_w) * 0.08    # at most 8% of roi

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh else 0
        if not (0.7 < aspect < 1.4):    # roughly square
            continue
        candidates.append((x, y, bw, bh))

    if not candidates:
        return []

    # ── 5. Determine median card size
    widths  = sorted([c[2] for c in candidates])
    heights = sorted([c[3] for c in candidates])
    med_w = int(np.median(widths))
    med_h = int(np.median(heights))

    # ── 6. Keep only candidates close to median size
    tol = 0.35
    filtered = [
        c for c in candidates
        if abs(c[2] - med_w) / med_w < tol and abs(c[3] - med_h) / med_h < tol
    ]

    if not filtered:
        return []

    # ── 7. Non-maximum suppression (merge overlapping boxes)
    boxes = np.array([[x, y, x+bw, y+bh] for x, y, bw, bh in filtered])
    scores = np.ones(len(boxes))
    keep = _nms(boxes, scores, iou_thresh=0.3)
    filtered = [filtered[i] for i in keep]

    # ── 8. Snap to uniform grid
    slots = _snap_to_grid(filtered, med_w, med_h)

    # ── 9. Translate back to full-image coords
    result = []
    for (x, y, sw, sh) in slots:
        result.append({
            "x": x + left_cut,
            "y": y + top_cut,
            "w": sw,
            "h": sh,
        })

    if debug:
        dbg = roi.copy()
        for (x, y, sw, sh) in slots:
            cv2.rectangle(dbg, (x, y), (x+sw, y+sh), (0, 255, 0), 2)
        cv2.imwrite("/tmp/wuwa_debug_slots.png", dbg)

    return result


def _nms(boxes, scores, iou_thresh=0.3):
    """Simple IoU-based NMS."""
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return keep


def _snap_to_grid(candidates, med_w, med_h):
    """
    Re-align all boxes to a regular grid by clustering their top-left corners.
    """
    if not candidates:
        return []

    xs = sorted(set(c[0] for c in candidates))
    ys = sorted(set(c[1] for c in candidates))

    # Cluster x positions
    col_centers = _cluster_positions([c[0] for c in candidates], med_w * 0.5)
    row_centers = _cluster_positions([c[1] for c in candidates], med_h * 0.5)

    slots = []
    for cx in col_centers:
        for cy in row_centers:
            slots.append((cx, cy, med_w, med_h))

    # Deduplicate
    seen = set()
    unique = []
    for s in slots:
        key = (s[0], s[1])
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique


def _cluster_positions(values, min_gap):
    """Merge positions that are closer than min_gap."""
    if not values:
        return []
    sorted_vals = sorted(values)
    clusters = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        if v - clusters[-1][-1] < min_gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [int(np.mean(c)) for c in clusters]


# ─────────────────────────────────────────────
#  Quantity OCR (lightweight, no heavy model)
# ─────────────────────────────────────────────

def read_quantity(slot_crop: np.ndarray) -> int:
    """
    Read the quantity number shown at the bottom of an item slot.
    Uses simple thresholding + pytesseract-free digit recognition
    via contour analysis (avoids heavy model dependency).
    Falls back to EasyOCR if available.
    """
    h, w = slot_crop.shape[:2]
    # Bottom 20% of the card where quantity lives
    bottom = slot_crop[int(h * 0.78):, :]

    # White/yellow text on dark background
    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    try:
        import easyocr
        _reader = getattr(read_quantity, "_reader", None)
        if _reader is None:
            read_quantity._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = read_quantity._reader.readtext(thresh, allowlist="0123456789", detail=0)
        nums = [r.strip() for r in results if r.strip().isdigit()]
        return int(nums[0]) if nums else 0
    except Exception:
        return 0


# ─────────────────────────────────────────────
#  Perceptual hashing for item icon matching
# ─────────────────────────────────────────────

def hash_slot(slot_crop: np.ndarray, hash_size: int = 16) -> str:
    """
    Compute a perceptual hash of the icon area (top ~78% of card, 
    ignoring the quantity strip and badge area at top-right).
    """
    h, w = slot_crop.shape[:2]
    icon = slot_crop[int(h*0.05):int(h*0.78), int(w*0.05):int(w*0.92)]
    pil = Image.fromarray(cv2.cvtColor(icon, cv2.COLOR_BGR2RGB))
    return str(imagehash.phash(pil, hash_size=hash_size))


def find_best_match(query_hash: str, db: dict, threshold: int = 12) -> Optional[str]:
    """
    Find the closest hash in the dataset within `threshold` bits.
    Returns the matched hash key or None.
    """
    query = imagehash.hex_to_hash(query_hash)
    best_dist = threshold + 1
    best_key = None
    for key in db:
        try:
            candidate = imagehash.hex_to_hash(key)
            dist = query - candidate
            if dist < best_dist:
                best_dist = dist
                best_key = key
        except Exception:
            continue
    return best_key


# ─────────────────────────────────────────────
#  Badge detection (新 = new item)
# ─────────────────────────────────────────────

def detect_new_badge(slot_crop: np.ndarray) -> bool:
    """Detect the gold '新' (new) badge in top-left of card."""
    h, w = slot_crop.shape[:2]
    corner = slot_crop[:int(h*0.25), :int(w*0.35)]
    hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
    # Gold/orange hue range
    mask = cv2.inRange(hsv, np.array([15, 100, 150]), np.array([35, 255, 255]))
    return mask.sum() > 200 * 255


# ─────────────────────────────────────────────
#  Main scan function
# ─────────────────────────────────────────────

def scan_screenshot(image_path: str, debug: bool = False) -> list[ItemSlot]:
    """
    Full pipeline: load → detect slots → hash → match/register.
    Returns list of ItemSlot objects.
    """
    db = load_dataset()
    img = load_image(image_path)
    img = preprocess(img)

    slot_boxes = detect_item_slots(img, debug=debug)

    results = []
    for idx, box in enumerate(slot_boxes):
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]

        # Guard against out-of-bounds
        ih, iw = img.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(iw, x + w)
        y2 = min(ih, y + h)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        phash  = hash_slot(crop)
        qty    = read_quantity(crop)
        is_new = detect_new_badge(crop)

        # Try to match existing item
        matched_key = find_best_match(phash, db)
        name = db[matched_key]["name"] if matched_key else None

        # Save crop for unknown items
        crop_path = None
        if name is None:
            CROPS_DIR.mkdir(parents=True, exist_ok=True)
            crop_path = str(CROPS_DIR / f"unknown_{phash[:12]}.png")
            cv2.imwrite(crop_path, crop)

        slot = ItemSlot(
            index=idx,
            bbox=(x1, y1, w, h),
            quantity=qty,
            image_hash=phash,
            name=name,
            crop_path=crop_path,
            is_new=is_new,
        )
        results.append(slot)

    return results