import cv2
import numpy as np
import os
from skimage import color
from skimage.filters import threshold_otsu

def load_and_preprocess_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (256, 256))  # 统一图片大小为256x256以便更好的处理
            images.append(img)
    return images

def locate_mangoes(images):
    all_locations = []
    for image in images:
        # 转换为灰度图像
        gray_image = color.rgb2gray(image)
        # 使用Otsu's threshold自动找到一个阈值
        thresh = threshold_otsu(gray_image)
        binary = gray_image > thresh
        # 使用轮廓检测来确定芒果的位置
        contours, _ = cv2.findContours((binary.astype(np.uint8)), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        locations = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # 可调整此阈值以筛选轮廓大小
                x, y, w, h = cv2.boundingRect(contour)
                locations.append((x, y, x+w, y+h))
        all_locations.append(locations)
    return all_locations

def mark_image(image, locations):
    for loc in locations:
        x1, y1, x2, y2 = loc
        # 在图片中绘制红色边框
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image

def save_marked_images(images, all_locations, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for idx, (image, locations) in enumerate(zip(images, all_locations)):
        marked_image = mark_image(image, locations)
        cv2.imwrite(os.path.join(folder, f'marked_image_{idx}.jpg'), marked_image)

# 使用实例
folder_path = r'mango/downloadfiles1/archive/data/shixunfiles/3a2b8256a18499c8b84a46b04cc63d8c_1705368625397'
output_folder = 'multi_output_test'
images = load_and_preprocess_images(folder_path)
all_locations = locate_mangoes(images)
save_marked_images(images, all_locations, output_folder)