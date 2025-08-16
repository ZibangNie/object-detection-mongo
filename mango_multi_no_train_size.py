import cv2
import numpy as np
import os

# 原有的红色和粉色范围
lower_red = np.array([0, 40, 40], dtype=np.uint8)
upper_red = np.array([30, 255, 255], dtype=np.uint8)

lower_pink = np.array([150, 40, 40], dtype=np.uint8)
upper_pink = np.array([180, 255, 255], dtype=np.uint8)

# 添加紫色范围
lower_purple = np.array([130, 55, 55], dtype=np.uint8)
upper_purple = np.array([190, 255, 255], dtype=np.uint8)

def enhance_red_pink_purple(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 创建红色、粉色和紫色的掩膜
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
    mask_pink = cv2.inRange(hsv_image, lower_pink, upper_pink)
    mask_purple = cv2.inRange(hsv_image, lower_purple, upper_purple)

    # 合并三个颜色的掩膜
    mask = cv2.bitwise_or(mask_red, mask_pink)
    mask = cv2.bitwise_or(mask, mask_purple)

    if cv2.countNonZero(mask) == 0:
        return image, mask

    # 增强图像的饱和度和亮度
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv[:, :, 1] = cv2.multiply(image_hsv[:, :, 1], 2)  # 增强饱和度
    image_hsv[:, :, 2] = cv2.add(image_hsv[:, :, 2], 50)      # 增加亮度

    # 将增强后的HSV图像转换回BGR图像
    enhanced_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    return enhanced_image, mask

def detect_ellipse_or_circle(contour, min_size):
    """判断一个轮廓是否为椭圆或圆形，并且检查椭圆的大小"""
    if len(contour) >= 5:
        # 尝试拟合椭圆
        ellipse = cv2.fitEllipse(contour)
        (center, axes, angle) = ellipse
        aspect_ratio = float(axes[0]) / axes[1]

        # 计算椭圆的面积
        area = np.pi * axes[0] * axes[1] / 4  # 椭圆的面积公式

        # 通过调整aspect_ratio的范围来适应更多的椭圆形状
        if 0.5 < aspect_ratio < 1.5 and area > min_size:  # 增加椭圆的最小面积限制
            return "ellipse", ellipse
    return None, None

def preprocess_and_detect_mango(image, min_ellipse_size):
    """预处理图像并检测可能的芒果区域"""
    enhanced_image, mask = enhance_red_pink_purple(image)

    if np.count_nonzero(mask) == 0:
        return enhanced_image, []

    # 应用形态学操作，去除噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    ellipses = []

    # 处理轮廓并判断形状
    for contour in contours:
        shape_type, shape = detect_ellipse_or_circle(contour, min_ellipse_size)

        if shape_type == "ellipse":
            center, axes, angle = shape
            center = tuple(map(int, center))
            axes = (int(axes[0] / 2), int(axes[1] / 2))  # 确保 axes 是整数

            # 绘制椭圆时，指定起始角度 startAngle 和结束角度 endAngle
            cv2.ellipse(enhanced_image, center, axes, angle, 0, 360, (255, 0, 0), 2)

            # 标记为芒果
            cv2.putText(enhanced_image, "Mango", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 处理紫色区域
    purple_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in purple_contours:
        shape_type, shape = detect_ellipse_or_circle(contour, min_ellipse_size)
        if shape_type == "ellipse":
            center, axes, angle = shape
            center = tuple(map(int, center))
            axes = (int(axes[0] / 2), int(axes[1] / 2))

            # 绘制椭圆时，指定起始角度 startAngle 和结束角度 endAngle
            cv2.ellipse(enhanced_image, center, axes, angle, 0, 360, (255, 0, 255), 2)

            # 标记为芒果
            cv2.putText(enhanced_image, "Purple Mango", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return enhanced_image, mask

# 创建输出文件夹
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

input_folder = 'mango/downloadfiles1/archive/data/shixunfiles/3a2b8256a18499c8b84a46b04cc63d8c_1705368625397'

min_ellipse_size = 500  # 设置最小椭圆面积阈值，只有大于此值的椭圆才会被标注

for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # 预处理图像并检测芒果
        enhanced_image, mask = preprocess_and_detect_mango(image, min_ellipse_size)

        # 保存结果图像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, enhanced_image)