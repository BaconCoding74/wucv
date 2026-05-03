# I do this project to study, so there will be a lot of comments
# There will be redundant code also because I feel more comfortable to learn 
# while isolating variables if possible in each step which is easier to revise
from PIL import Image
import imagehash
import cv2
import os
import itertools
import numpy as np


# DEVELOPMENT
GET_DIGIT = True

# Filtration condition
MIN_AREA  = 15900
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
ICON_SIZE = 128
MIN_ACCEPTABLE_SIMILARITY = 0.7 
MAX_SIMILARITY = 0.999

item_icon_folder = 'items_assets/icons'
resource_folder = 'test_cases'
target = "wuwa_inventory_system_20.png"
target_path = f'{resource_folder}/{target}'

# Create debug folder if not exist
debug_folder = 'debug'
tc_folder = f'{debug_folder}/{target}'
os.makedirs(tc_folder, exist_ok=True)

items = os.listdir(tc_folder)
attempt_folder = f'{tc_folder}/attempt_{len(items) + 1}'
os.makedirs(attempt_folder, exist_ok=True)

# Normal read image
img = cv2.imread(target_path)

# Scaling down, INTER_AREA allow downscaling with less distortion
img = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

# Reduce complexity by converting to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(f'{attempt_folder}/1_gray.png', gray)

# Use Canny edge detection to find edges in the 
# Threshold 1 is threshold that used to identify weak edge through pixel gradient
# Weak edge may kept if it connect strong edge, otherwise it will be discarded
# Threshold 2 is threshold that used to identify strong edge through pixel gradient
edges = cv2.Canny(gray, 30, 100)
cv2.imwrite(f'{attempt_folder}/2_edges.png', edges)

# Create a kernel, it define the shape for dilation and erosion
# In this case, it means that pixel will only dilate or erode neighboring pixel within 3x3 rectangle
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 7))

# Dilate means grow, erode means shrink
# Dilate the edges to make them more connected, which can help in finding contours
edges = cv2.dilate(edges, dilate_kernel, iterations=1)
cv2.imwrite(f'{attempt_folder}/3_dilated_edges.png', edges)

# Three step below is to eliminate diagonal lines which distort the result
# extract horizontal lines that only have rectangle with width of 40 pixel
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
cv2.imwrite(f'{attempt_folder}/4_horizontal_lines.png', horizontal_lines)

# extract vertical lines that only have rectangle with height of 40 pixel
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
cv2.imwrite(f'{attempt_folder}/5_vertical_lines.png', vertical_lines)

# combine only card-like straight lines
card_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
cv2.imwrite(f'{attempt_folder}/6_card_lines.png', card_lines)

# Contours is curve that joins all continuous points
# That means it is a shape or boundary of an object
# RETR_EXTERNAL means find outermost boundaries
# CHAIN_APPROX_SIMPLE means compress continous points which means it only store key points
contours, _ = cv2.findContours(card_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Visualize contours
backup_img = img.copy()
for c in contours:
    cv2.drawContours(backup_img, c, -1, (0,255,0), 2)

cv2.imwrite(f'{attempt_folder}/7_contours.png', backup_img)

# The visualization of contours only draw key points of boundary of objects
# We need to get rectangle bounding box for item
backup_img = img.copy()
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(backup_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imwrite(f'{attempt_folder}/8_rectangle_bounding_boxes.png', backup_img)

# From the rectangle bounding boxes, it shows selection of items including noises
# Now we check the area to see if it can be used for filtering
backup_img = img.copy()
for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    # Calculate area
    area = w * h

    cv2.rectangle(backup_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(backup_img, f'area: {area}', (x, y + TEXT_Y_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# From observation, it show that item grid area > 10000
cv2.imwrite(f'{attempt_folder}/9_area.png', backup_img)

# From the rectangle bounding boxes, it shows selection of items including noises
# Now we check the ratio to see if it can be used for filtering
backup_img = img.copy()
for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    # Calculate ratio
    ratio = w / h

    cv2.rectangle(backup_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(backup_img, f'ratio: {ratio:.4f}', (x, y + TEXT_Y_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# From observation, it show that item grid ratio is around 0.8 to 0.9
cv2.imwrite(f'{attempt_folder}/10_ratio.png', backup_img)

# Filtering by area and visualize the boxes
backup_img = img.copy()
cv2.putText(backup_img, f'Filter by area', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    area = w * h
    if area < MIN_AREA:
        continue

    cv2.rectangle(backup_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imwrite(f'{attempt_folder}/11_filtered_by_area.png', backup_img)

# Filtering by ratio and visualize the boxes
backup_img = img.copy()
cv2.putText(backup_img, f'Filter by ratio', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    ratio = w / h
    if ratio < MIN_RATIO or ratio > MAX_RATIO:
        continue

    cv2.rectangle(backup_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imwrite(f'{attempt_folder}/12_filtered_by_ratio.png', backup_img)

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

cv2.imwrite(f'{attempt_folder}/13_filtered_by_area_and_ratio.png', backup_img)

if (len(boxes) == 0):
    print("No box detected after filtration, please adjust the filtration condition")
    exit()

# Sort boxes by row then column
semi_y_sorted_boxes = sorted(boxes, key=lambda b: b[1])
row = []
row_num = 0
previous_y = float('-inf')

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

sorted_nested_boxes = [sorted(r, key=lambda b: b[0]) for r in row]
sorted_boxes = itertools.chain.from_iterable(sorted_nested_boxes)

# Stage 2: Image recognition and get number of items from selected boxes
ICON_HEIGHT_RATIO = 0.8
for i, box in enumerate(sorted_boxes):
    x, y, w, h = box
    card = img[y:y+h, x:x+w]

    # Check if icon and quantity is correctly obtained
    icon_img = card[0:int(h*ICON_HEIGHT_RATIO), 0:w]
    cv2.imwrite(f'{attempt_folder}/14_icon_{i}.png', icon_img)

    quantity_img = card[int(h*ICON_HEIGHT_RATIO):h, int(w/2):w]
    cv2.imwrite(f'{attempt_folder}/15_quantity_{i}.png', quantity_img)

    # Stage 3: Template matching to get quantity
    # Eliminate color information since it is not required for text recognition
    gray_quantity = cv2.cvtColor(quantity_img, cv2.COLOR_BGR2GRAY)

    # Resize to make the size of number larger
    enlarged_quantity = cv2.resize(gray_quantity, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

    # Dilation to prrevent digit broken into pieces
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_quantity = cv2.dilate(enlarged_quantity, dilate_kernel, iterations=1)
    cv2.imwrite(f'{attempt_folder}/16_dilated_quantity_{i}.png', dilated_quantity)
    
    # Thresholding to make number more contrast
    # THRESH_BINARY is default which change pixel to predefined maxval if pixel value > threshold
    # Set 180 as threshold, if pixel value > 180 to 255, otherwise set to 0
    # In this case, the number is white, so maxval is set to 255 (white)
    _, thresh = cv2.threshold(dilated_quantity, THRESHOLD_VALUE, MAX_VALUE, cv2.THRESH_BINARY)

    # Crop out the white card boundary which distort the digit segmentation process
    H_thresh, W_thresh = thresh.shape[:2]
    thresh = thresh[0:int(H_thresh * TOP_REMAIN_RATIO), 0:int(W_thresh * LEFT_REMAIN_RATIO)]
    cv2.imwrite(f'{attempt_folder}/17_thresholded_quantity_{i}.png', thresh)

    # Get digit image for template matching
    digit_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sorting contour by x so that the digit is in correct order
    digit_boxes = [cv2.boundingRect(c) for c in digit_contours]
    digit_boxes = sorted(digit_boxes, key = lambda b: b[0])

    templates = {}
    for digit in range(10):
        template_image = cv2.imread(f'digit_templates/{digit}.png', cv2.IMREAD_GRAYSCALE)
        templates[str(digit)] = template_image

    result_digit = ''
    preprocessed_digit_boxes = []
    for j, box in (enumerate(digit_boxes)):
        x, y, w, h = box

        digit_img = thresh[y:y+h, x:x+w]
        cv2.imwrite(f'{attempt_folder}/18_digit_{i}_{j}.png', digit_img)

        downscaled = cv2.resize(digit_img, (24, 32), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'{attempt_folder}/19_downscaled_digit_{i}_{j}.png', downscaled)
        preprocessed_digit_boxes.append(downscaled)

    for j, digit_box in enumerate(preprocessed_digit_boxes):
        best_match = None
        best_score = float('-inf')

        for digit, template in templates.items():
            # CCOEFF means correlation coefficient, it measure the similarity of two images
            # NORMED means normalized
            score = cv2.matchTemplate(digit_box, template, cv2.TM_CCOEFF_NORMED)[0][0]

            if score > best_score:
                best_score = score
                best_match = digit
            

        result_digit += best_match
        cv2.imwrite(f'{attempt_folder}/20_matched_digit_{i}_{j}_{best_match}.png', digit_box)
    
    # print(f"Result digit for box {i}: {result_digit}")

    # Step 4: Item icon recognition
    # Convery BGR format (default OpenCV format) to RGB format (default PIL format)
    icon_img_scaled = cv2.resize(icon_img, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA)
    icon_img_gray = cv2.cvtColor(icon_img_scaled, cv2.COLOR_BGR2GRAY)
    icon_img_edges = cv2.Canny(icon_img_gray, 30, 100)

    ih, iw = icon_img_edges.shape

    clean = np.zeros_like(icon_img_edges)

    for (y, x) in zip(*np.where(icon_img_edges > 0)):
        # remove pixels near border
        if x < 5 or x > iw-8 or y < 10 or y > ih-25:
            continue

        clean[y, x] = 255

    icon_img_contour, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    icon_clone = icon_img_scaled.copy()

    icon_img_contour = [c for c in icon_img_contour if cv2.arcLength(c, False) > 10]
    all_points = np.vstack(icon_img_contour)
    x, y, w, h = cv2.boundingRect(all_points)
    
    item_only = icon_clone[y:y+h, x:x+w]
    resized_item = cv2.resize(item_only, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA)
    
    cv2.imwrite(f'{attempt_folder}/21_icon_{i}_with_contours.png', resized_item)
    cv2.imwrite(f'{attempt_folder}/21_icon_clean_boundary_{i}.png', clean)

    # Perpetual hashing
    icon_files = os.listdir(f'{item_icon_folder}')
    item_id = None
    highest_similarity = float('-inf')
    for j, file in enumerate(icon_files):
        template_icon = cv2.imread(f'{item_icon_folder}/{file}')

        score = cv2.matchTemplate(resized_item, template_icon, cv2.TM_CCOEFF_NORMED)[0][0]
        print(f'{i} vs {file}, score: {score}')

    
    # If not found or there is new image, save the icon image
    # item_name = item_id if item_id is not None else f'unknown_{len(icon_files) + 1}'
    # print(f'{item_name}: {result_digit}')

    # if highest_similarity >= MAX_SIMILARITY:
    #     continue

    # if item_id is None:
    #     print(f'NOT FOUND: box_{i}, new icon named {item_name} created with similarity {highest_similarity:.4f}')
    # else:
    #     print(f'SIMILAR ICON DETECTED: box_{i} is similar to {item_name} with similarity {highest_similarity:.4f}')
    
    # cv2.imwrite(f'{item_icon_folder}/{len(icon_files)}_t.png', resized_item)
        
    
    # print(f'Item ID: {item_name}, quantity: {result_digit}')
        
