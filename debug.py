# I do this project to study, so there will be a lot of comments
# There will be redundant code also because I feel more comfortable to learn 
# while isolating variables if possible in each step which is easier to revise
import cv2
import numpy as np
from enum import Enum

# Filtration condition
MIN_AREA  = 10000
MIN_RATIO = 0.777
MAX_RATIO = 0.87

# Normal read image
img = cv2.imread('resources/wuwa_inventory_system_15.png')

# Scaling down, INTER_AREA allow downscaling with less distortion
img = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

# Reduce complexity by converting to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Use Canny edge detection to find edges in the image
edges = cv2.Canny(gray, 40, 120)
cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create a kernel, it define the shape for dilation and erosion
# In this case, it means that pixel will only dilate or erode neighboring pixel within 3x3 rectangle
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 7))

# Dilate means grow, erode means shrink
# Dilate the edges to make them more connected, which can help in finding contours
edges = cv2.dilate(edges, dilate_kernel, iterations=1)
cv2.imshow('dilated edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Three step below is to eliminate diagonal lines which distort the result
# extract horizontal lines that only have rectangle with width of 40 pixel
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
cv2.imshow('horizontal lines', horizontal_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()

# extract vertical lines that only have rectangle with height of 40 pixel
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
cv2.imshow('vertical lines', vertical_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()

# combine only card-like straight lines
card_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
cv2.imshow('card-like lines', card_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Contours is curve that joins all continuous points
# That means it is a shape or boundary of an object
# RETR_EXTERNAL means find outermost boundaries
# CHAIN_APPROX_SIMPLE means compress continous points which means it only store key points
contours, _ = cv2.findContours(card_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Visualize contours
backup_img = img.copy()
for c in contours:
    cv2.drawContours(backup_img, c, -1, (0,255,0), 2)

cv2.imshow('contours', backup_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# The visualization of contours only draw key points of boundary of objects
# We need to get rectangle bounding box for item
backup_img = img.copy()
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(backup_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('rectangle bounding boxes', backup_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# From the rectangle bounding boxes, it shows selection of items including noises
# Now we check the area to see if it can be used for filtering
backup_img = img.copy()
for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    # Calculate area
    area = w * h

    cv2.rectangle(backup_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(backup_img, f'area: {area}', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# From observation, it show that item grid area > 10000
cv2.imshow('area', backup_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# From the rectangle bounding boxes, it shows selection of items including noises
# Now we check the ratio to see if it can be used for filtering
backup_img = img.copy()
for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    # Calculate ratio
    ratio = w / h

    cv2.rectangle(backup_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(backup_img, f'ratio: {ratio}', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# From observation, it show that item grid ratio is around 0.8 to 0.9
cv2.imshow('ratio', backup_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Filtering by area and visualize the boxes
backup_img = img.copy()
cv2.putText(backup_img, f'Filter by area', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    area = w * h
    if area < MIN_AREA:
        continue

    cv2.rectangle(backup_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('filtration_area', backup_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Filtering by ratio and visualize the boxes
backup_img = img.copy()
cv2.putText(backup_img, f'Filter by ratio', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    ratio = w / h
    if ratio < MIN_RATIO or ratio > MAX_RATIO:
        continue

    cv2.rectangle(backup_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('filtration_ratio', backup_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Filtering by area and ratio and visualize the boxes
boxes = []
backup_img = img.copy()
cv2.putText(backup_img, f'Filter by area and ratio', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    area = w * h
    if area <= MIN_AREA:
        continue

    ratio = w / h
    if ratio < MIN_RATIO or ratio > MAX_RATIO:
        continue

    boxes.append((x, y, w, h))
    cv2.rectangle(backup_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('filtration_final', backup_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



