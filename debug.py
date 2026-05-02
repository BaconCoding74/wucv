# I do this project to study, so there will be a lot of comments
# There will be redundant code also because I feel more comfortable to learn 
# while isolating variables if possible in each step which is easier to revise
import cv2
import os
import easyocr
import pytesseract

# Filtration condition
MIN_AREA  = 19900
MIN_RATIO = 0.773
MAX_RATIO = 0.87

# Text position config
Y_OFFSET = 15

resource_folder = 'resources'
target = "wuwa_inventory_system_9.png"
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

# Use Canny edge detection to find edges in the image
edges = cv2.Canny(gray, 40, 120)
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
    cv2.putText(backup_img, f'area: {area}', (x, y + Y_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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
    cv2.putText(backup_img, f'ratio: {ratio:.4f}', (x, y + Y_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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

# Stage 2: Image recognition and get number of items from selected boxes
ICON_HEIGHT_RATIO = 0.8
for i, box in enumerate(boxes):
    x, y, w, h = box
    card = img[y:y+h, x:x+w]

    # Check if icon and quantity is correctly obtained
    icon_img = card[0:int(h*ICON_HEIGHT_RATIO), 0:w]
    cv2.imwrite(f'{attempt_folder}/14_icon_{i}.png', icon_img)

    quantity_img = card[int(h*ICON_HEIGHT_RATIO):h, int(w/2):w]
    cv2.imwrite(f'{attempt_folder}/15_quantity_{i}.png', quantity_img)

    # Stage 3: OCR to get quantity
    # Eliminate color information since it is not required for text recognition
    gray_quantity = cv2.cvtColor(quantity_img, cv2.COLOR_BGR2GRAY)

    # Resize to make the size of number larger
    enlarged_quantity = cv2.resize(gray_quantity, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

    blurred_quantity = cv2.GaussianBlur(enlarged_quantity, (5, 5), 0)
    
    # Thresholding to make number more contrast
    # THRESH_BINARY is default which change pixel to predefined maxval if pixel value > threshold
    # Set 180 as threshold, if pixel value > 180 to 255, otherwise set to 0
    # In this case, the number is white, so maxval is set to 255 (white)
    _, thresh = cv2.threshold(blurred_quantity, 180, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f'{attempt_folder}/16_thresholded_quantity_{i}.png', thresh)

    digit_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in digit_contours:
        cv2.drawContours(thresh, c, -1, (0, 255, 0), 2)
    cv2.imwrite(f'{attempt_folder}/17_digit_contours_{i}.png', thresh)

    # Use easyocr to read number from thresholded image
    # detail = 1 means returning bonding box, confidence and text else, only return text
    # reader = easyocr.Reader(['en'])
    # result = reader.readtext(thresh, allowlist='01234567890', detail=0)
    # print(i, result)

    config = '--psm 7 -c tessedit_char_whitelist=0123456789'
    result = pytesseract.image_to_string(thresh, config=config)
    print(i, result.strip())