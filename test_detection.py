# I do this project to study, so there will be a lot of comments
# There will be redundant code also because I feel more comfortable to learn 
# while isolating variables if possible in each step which is easier to revise
from PIL import Image
from pathlib import Path
import cv2
import os
import itertools
from cv2.typing import MatLike, Rect
from dataclasses import dataclass
from benchmarking_context import benchmark
from image_recognition.dinov2_embedding import ImageRetriever
from skimage.metrics import structural_similarity as ssim
from enum import Enum
import numpy as np


# DEVELOPMENT
GET_DIGIT = True

# Image Scale
IMAGE_SCALE = 0.6

# Filtration condition
MIN_AREA  = 15900
MIN_AREA_RATIO = 0.5
MAX_AREA_RATIO = 1.5
MIN_RATIO = 0.773
MAX_RATIO = 0.87

# Text position config
TEXT_Y_OFFSET = 15

# Thresholding config
THRESHOLD_VALUE = 180
MAX_VALUE = 255
LEFT_REMAIN_RATIO = 0.87
TOP_REMAIN_RATIO = 0.79

# Image recognition config
ICON_HEIGHT_RATIO = 0.8
ICON_SIZE = 128
MIN_ACCEPTABLE_SIMILARITY = 0.7 
MAX_SIMILARITY = 0.999
NEW_INDICATOR_MASK_RATIO = 0.235, 0.781

item_icon_folder = 'items_assets/icons'
resource_folder = 'test_cases'

class ItemMatchType(Enum):
    EXISTING = 1
    NEW = 2

def load_templates() -> dict[str, MatLike]:
    templates = {}
    for digit in range(10):
        template_path = f'digit_templates/{digit}.png'
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        templates[str(digit)] = template_img
    return templates

def detect_item_boxes(image: MatLike) -> list[Rect]:
    # Reduce complexity by converting to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('temp/gray.png', gray)

    # Find edges
    edges = cv2.Canny(gray, 30, 100)

    # Dilate the edges to make them more connected, which can help in finding contours
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 7))
    edges = cv2.dilate(edges, dilate_kernel, iterations=1)

    # Remove background noise and only keep the horizontal and vertical lines which represent grid lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)

    card_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # Find contours of the card lines
    contours, _ = cv2.findContours(card_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtering by area and ratio and visualize the boxes
    boxes = []
    backup_img = image.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        area = w * h
        if area <= MIN_AREA:
            continue

        ratio = w / h
        if ratio < MIN_RATIO or ratio > MAX_RATIO:
            continue

        cv2.rectangle(backup_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(backup_img, f'area: {area}', (x, y + TEXT_Y_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        boxes.append((x, y, w, h))

    cv2.imwrite(f'temp/item_boxes.png', backup_img)

    # Filter out boxes with abnormal sizes
    areas = [w * h for (x, y, w, h) in boxes]
    median_area = np.median(areas)
    filtered_boxes = [
        box for areas, box in zip(areas, boxes) 
        if MIN_AREA_RATIO * median_area <= areas <= MAX_AREA_RATIO * median_area
    ]

    return filtered_boxes

def sort_item_boxes(boxes: list[Rect]) -> list[Rect]:
    # Initial sort with y
    semi_y_sorted_boxes = sorted(boxes, key=lambda b: b[1])
    row = []
    row_num = 0
    previous_y = float('-inf')

    # Solve minor y difference between boxes in the same row
    for box in semi_y_sorted_boxes:
        x, y, w, h = box

        if (previous_y == float('-inf')):
            row.append([box])
            previous_y = y
        elif(abs(y - previous_y) < h / 2):
            row[row_num].append(box)
            previous_y = y
        else:
            row_num += 1
            row.append([box])
            previous_y = y

    # Sort each row by x
    sorted_nested_boxes = [sorted(r, key=lambda b: b[0]) for r in row]
    return list(itertools.chain.from_iterable(sorted_nested_boxes))

def split_item_card(card: MatLike) -> tuple[MatLike, MatLike]:
    h, w = card.shape[:2]

    icon_img = card[:int(h*ICON_HEIGHT_RATIO), :]
    quantity_img = card[int(h*ICON_HEIGHT_RATIO):, int(w/2):]

    return icon_img, quantity_img

def imwrite_increment(path: str, image):
    path = Path(path)

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    parent.mkdir(parents=True, exist_ok=True)

    candidate = path
    i = 1

    while candidate.exists():
        candidate = parent / f"{stem}_{i}{suffix}"
        i += 1

    cv2.imwrite(str(candidate), image)
    return candidate

def get_quantity_from_image(quantity_img: MatLike, templates: dict[str, MatLike], index: int) -> str:
    # Eliminate color information since it is not required for text recognition
    gray_quantity = cv2.cvtColor(quantity_img, cv2.COLOR_BGR2GRAY)

    # Resize to make the size of number larger for better recognition
    enlarged_quantity = cv2.resize(gray_quantity, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

    # Dilation to prrevent digit broken into pieces
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_quantity = cv2.dilate(enlarged_quantity, dilate_kernel, iterations=1)
    cv2.imwrite(f'temp/15_dilated_quantity_{index}.png', dilated_quantity)

    # Change to binary image
    # Set value that greater or equal to THRESHOLD_VALUE to MAX_VALUE, otherwise set to 0
    _, thresh = cv2.threshold(dilated_quantity, THRESHOLD_VALUE, MAX_VALUE, cv2.THRESH_BINARY)

    # Crop out the white card boundary which distort the digit segmentation process
    H_thresh, W_thresh = thresh.shape[:2]
    thresh = thresh[0:int(H_thresh * TOP_REMAIN_RATIO), 0:int(W_thresh * LEFT_REMAIN_RATIO)]
    cv2.imwrite(f'temp/16_quantity_{index}.png', thresh)

    # Get digit image for template matching
    digit_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangle bounding boxes for each digit contour
    digit_boxes = [cv2.boundingRect(c) for c in digit_contours]

    thresh_h, thresh_w = thresh.shape[:2]

    # Remark: Need to make it to constant later after debugging :D
    min_digit_box_w = int(thresh_w * 0.08)
    max_digit_box_w = int(thresh_w * 0.4)
    min_digit_box_h = int(thresh_h * 0.5)
    max_digit_box_h = int(thresh_h * 0.9)
    digit_boxes = [
        b for b in digit_boxes 
        if (b[2] >= min_digit_box_w and b[2] <= max_digit_box_w) and (b[3] >= min_digit_box_h and b[3] <= max_digit_box_h)
    ]

    # Sort the digit boxes by their x-coordinate
    digit_boxes = sorted(digit_boxes, key = lambda b: b[0])

    result_digit = ''
    for j, (x, y, w, h) in enumerate(digit_boxes):
        digit_img = thresh[y:y+h, x:x+w]
        downscaled_digit = cv2.resize(digit_img, (24, 32), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'temp/17_digit_{index}_{j}.png', downscaled_digit)
        
        best_match = None
        best_score = float('-inf')

        for digit, template in templates.items():
            # CCOEFF means correlation coefficient, it measure the similarity of two images
            # NORMED means normalized
            score = cv2.matchTemplate(downscaled_digit, template, cv2.TM_CCOEFF_NORMED)[0][0]

            if score > best_score:
                best_score = score
                best_match = digit

        result_digit += best_match
    
    return result_digit

def check_item_existence(icon_imgs: list[MatLike], retriever: ImageRetriever) -> list[dict]:
    if retriever.embedding_store.is_empty:
        return [{"match_type": ItemMatchType.NEW, "score": 1.0, "retrieved_path": None} for _ in icon_imgs]

    with benchmark("Image retrieval"):
        retrieved_results = retriever.retrieve_top_k(icon_imgs, k=5)

    processed_results = []

    for icon_img, retrieved_list in zip(icon_imgs, retrieved_results):
        most_relevant_retrieved = retrieved_list[0]

        if most_relevant_retrieved["score"] >= 0.95:
            most_relevant_retrieved["match_type"] = ItemMatchType.EXISTING
            processed_results.append(most_relevant_retrieved)
            continue

        query_image = cv2.resize(icon_img, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA)
        retrieved_image = cv2.imread(str(retriever.embedding_store.input_dir / most_relevant_retrieved["retrieved_path"]))
        retrieved_image = cv2.resize(retrieved_image, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA)

        # Sensitive to pixel alignment and geometry
        ssim_score, ssim_map = ssim(query_image, retrieved_image, channel_axis=-1, data_range=255, full=True)

        # Pooling channel score
        if ssim_map.ndim == 3:
            ssim_map = ssim_map.mean(axis=-1)

        # Create masking for new indicator
        h, w = ssim_map.shape[:2]
        valid_mask = np.ones((h, w), dtype=bool)
        valid_mask[:int(h * NEW_INDICATOR_MASK_RATIO[0]), int(w * NEW_INDICATOR_MASK_RATIO[1]):] = False

        # Calculate masked SSIM score
        masked_ssim_score = ssim_map[valid_mask].mean()
        if most_relevant_retrieved["score"] >= 0.91 and masked_ssim_score >= 0.84:
            most_relevant_retrieved["match_type"] = ItemMatchType.EXISTING
        else:
            print(f"Item is considered new based on SSIM score: {masked_ssim_score:.4f} and retrieval score: {most_relevant_retrieved['score']:.4f} and relevant retrieval path: {most_relevant_retrieved['retrieved_path']}")
            most_relevant_retrieved["match_type"] = ItemMatchType.NEW

        processed_results.append(most_relevant_retrieved)

    return processed_results

def get_next_item_id(folder_path: Path) -> int:
    existing_ids = []
    for f in folder_path.glob("item_*.png"):
        digit_part = f.stem.split('_')[-1]

        if not digit_part.isdigit():
            continue

        existing_ids.append(int(digit_part))

    return max(existing_ids, default=0) + 1

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int

    label: str
    amount: int

class ItemDetector:
    def __init__(self, retriever: ImageRetriever):
        self.retriever = retriever
        self.previous_detections = []
        self.previous_frame = None
        self.digit_templates = load_templates()
        self.detection_counter = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        new_items = any(detection.label == "NEW" for detection in self.previous_detections)
        if self.previous_frame is not None and not new_items:
            frame_diff = cv2.absdiff(frame, self.previous_frame)

            if frame_diff.mean() < 1.0:
                return self.previous_detections

        self.detection_counter += 1
        print(f"\nDetection attempt {self.detection_counter}:")

        with benchmark("Item box detection"):
            boxes = detect_item_boxes(frame)

        if (len(boxes) == 0):
            print("No box detected after filtration, please adjust the filtration condition")
            return [], [], []

        sorted_boxes = sort_item_boxes(boxes)
        result_digits = []
        icon_imgs = []

        with benchmark("Item processing"):
            for i, (x, y, w, h) in enumerate(sorted_boxes):
                card = frame[y:y+h, x:x+w]

                # Check if icon and quantity is correctly obtained
                icon_img, quantity_img = split_item_card(card)

                # Stage 3: Template matching to get quantity
                if GET_DIGIT:
                    result_digit = get_quantity_from_image(quantity_img, self.digit_templates, i)

                    if result_digit == '':
                        print(f"Failed to detect quantity for box {i}")
                        result_digit = '-1'

                    result_digits.append(int(result_digit))

                icon_imgs.append(icon_img)

        # Stage 4: Image recognition to get item name
        image_retriever = self.retriever
        item_infos = check_item_existence(icon_imgs, image_retriever)
        for item_info, icon_img in zip(item_infos, icon_imgs):
            if item_info["match_type"] == ItemMatchType.NEW:
                new_item_id = get_next_item_id(image_retriever.embedding_store.input_dir)
                print(f"New item detected, saving icon as item_{new_item_id}.png and adding to embedding store")

                # Save the new item icon to the embedding store input directory
                cv2.imwrite(f'{image_retriever.embedding_store.input_dir}/item_{new_item_id}.png', icon_img)
                image_retriever.embedding_store.sync(batch_size=32, force_hash_check=False, rebuild_on_mismatch=True)
                image_retriever.refresh()

        detections = []
        for (x, y, width, height), digit, item in zip(sorted_boxes, result_digits, item_infos):
            detections.append(
                Detection(
                    x1=x,
                    y1=y,
                    x2=x + width,
                    y2=y + height,
                    label="NEW" if item["match_type"] == ItemMatchType.NEW else "EXI",
                    amount=digit,
                )
            )

        self.previous_frame = frame.copy()
        self.previous_detections = detections

        return detections

        
            
