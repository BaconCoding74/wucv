from PIL import Image
import cv2

# Get new indicator in the image
# image = cv2.imread("image_recognition/datasets/datasets_v002/plates_1/tc_wuwa_inventory_system_9.png_1.png")
image = cv2.imread("image_recognition/datasets/datasets_v002/cube_1/tc_wuwa_inventory_system_9.png_1.png")
image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
h, w = image.shape[:2]
# cv2.imshow("Image", image[:30, 100:])

# Relative coordinates
cv2.imshow("Image", image[:int(h * 0.235), int(w * 0.781):])
cv2.waitKey(0)
cv2.destroyAllWindows()