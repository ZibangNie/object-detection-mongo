import cv2
import numpy as np
import os


def segment_mango_by_color(image):
    # 将图像从BGR转换到HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 设置芒果颜色的HSV阈值范围

    lower_hsv = np.array([20, 100, 100])
    upper_hsv = np.array([30, 255, 255])

    # 创建掩码，只保留指定HSV范围内的颜色
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    # 可选：使用形态学操作改善掩码
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def load_images(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames


# 示例用法
folder_path = r'mango/downloadfiles/archive/data/shixunfiles/f38d473718e2aefcd940898e643fd0cd_1704251553388'  # 数据集文件夹路径
images, filenames = load_images(folder_path)
if not images:
    print("No images loaded.")
else:
    for img, filename in zip(images, filenames):
        mask = segment_mango_by_color(img)
        # 显示图像和掩码
        cv2.imshow('Image', img)
        cv2.imshow('Mask', mask)
        cv2.waitKey(0)  # 等待按键后继续
        cv2.destroyAllWindows()