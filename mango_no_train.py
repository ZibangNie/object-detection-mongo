import cv2
import numpy as np
import os

# 定义颜色阈值范围
lower_green = np.array([40, 100, 100], dtype=np.uint8)
upper_green = np.array([90, 255, 255], dtype=np.uint8)

# 扩大粉色阈值范围以捕捉淡粉色
lower_pink = np.array([130, 50, 50], dtype=np.uint8)  # 更低的H值以包含淡粉色
upper_pink = np.array([180, 255, 255], dtype=np.uint8)

lower_purple_hue_1 = np.array([120, 40, 40], dtype=np.uint8)
upper_purple_hue_1 = np.array([140, 255, 255], dtype=np.uint8)
lower_purple_hue_2 = np.array([0, 40, 40], dtype=np.uint8)
upper_purple_hue_2 = np.array([30, 255, 255], dtype=np.uint8)


def create_purple_mask(hsv_image):
    mask_purple_1 = cv2.inRange(hsv_image, lower_purple_hue_1, upper_purple_hue_1)
    mask_purple_2 = cv2.inRange(hsv_image, lower_purple_hue_2, upper_purple_hue_2)
    return cv2.bitwise_or(mask_purple_1, mask_purple_2)


def is_mango_contour(contour, area_threshold=500, circularity_threshold=0.5):
    area = cv2.contourArea(contour)
    if area < area_threshold:
        return False
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.04 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) <= 6:  # 接近圆形或椭圆形
        return True
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    circularity = hull_area / area
    return circularity > circularity_threshold


output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

input_folder = 'mango/downloadfiles1/archive/data/shixunfiles/3a2b8256a18499c8b84a46b04cc63d8c_1705368625397'
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
        mask_pink = cv2.inRange(hsv_image, lower_pink, upper_pink)
        mask_purple = create_purple_mask(hsv_image)

        mask = cv2.bitwise_or(cv2.bitwise_or(mask_green, mask_pink), mask_purple)

        # 应用CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv_image_clahe = cv2.cvtColor(clahe.apply(image), cv2.COLOR_BGR2HSV)
        mask_clahe = cv2.inRange(hsv_image_clahe, lower_pink, upper_pink)  # 重新应用阈值以利用增强的图像
        mask = cv2.bitwise_or(mask, mask_clahe)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if is_mango_contour(contour):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)