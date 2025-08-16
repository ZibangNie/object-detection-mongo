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
    locations = []
    for image in images:
        # 转换为灰度图像
        gray_image = color.rgb2gray(image)
        # 使用Otsu's threshold自动找到一个阈值
        thresh = threshold_otsu(gray_image)
        binary = gray_image > thresh
        # 使用轮廓检测来确定芒果的位置
        contours, _ = cv2.findContours((binary.astype(np.uint8)), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 假设最大的轮廓为芒果
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        locations.append((x, y, x+w, y+h))
    return locations

def mark_image(image, location):
    x1, y1, x2, y2 = location
    # 在图片中绘制红色边框
    return cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

def save_marked_images(images, locations, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for idx, (image, loc) in enumerate(zip(images, locations)):
        marked_image = mark_image(image, loc)
        cv2.imwrite(os.path.join(folder, f'marked_image_{idx}.jpg'), marked_image)

# 使用实例
folder_path_train = 'mango\\downloadfiles\\archive\\data\\shixunfiles\\f38d473718e2aefcd940898e643fd0cd_1704251553388'
folder_path_test='mango\\downloadfiles1\\archive\\data\\shixunfiles\\3a2b8256a18499c8b84a46b04cc63d8c_1705368625397'
output_folder_train = 'output_train'
output_folder_test = 'output_test'
images = load_and_preprocess_images(folder_path_train)
locations = locate_mangoes(images)
save_marked_images(images, locations, output_folder_train)
images = load_and_preprocess_images(folder_path_test)
locations = locate_mangoes(images)
save_marked_images(images, locations, output_folder_test)