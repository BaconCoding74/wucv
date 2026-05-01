import cv2
import numpy as np

img = cv2.imread('resources/wuwa_inventory_system_1.png')
H, W = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 40, 120)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
edges = cv2.dilate(edges, kernel, iterations=1)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []

for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    area = w * h

    if area < 3000:
        continue

    # rough shape filter
    ratio = w / h
    if 0.6 <= ratio <= 1.1:
        boxes.append((x, y, w, h))

widths = [b[2] for b in boxes]
heights = [b[3] for b in boxes]

median_w = np.median(widths)
median_h = np.median(heights)

valid_boxes = []

for x, y, w, h in boxes:
    if abs(w - median_w) > median_w * 0.25:
        continue

    if abs(h - median_h) > median_h * 0.25:
        continue

    valid_boxes.append((x, y, w, h))

final_boxes = []

region_h, region_w = img.shape[:2]

for x, y, w, h in valid_boxes:
    if y <= 5:
        continue

    if y + h >= region_h - 5:
        continue

    final_boxes.append((x, y, w, h))

for x, y, w, h in final_boxes:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.imshow('edges', img)
cv2.waitKey(0)
cv2.destroyAllWindows()