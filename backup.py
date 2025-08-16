import cv2
import numpy as np
import os


def preprocess_images(input_folder, output_folder, threshold_value=127, morph_kernel_size=(5, 5),
                      dilate_kernel_size=(3, 3),
                      erode_kernel_size=(3, 3), lower_green=None, upper_green=None):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    contour_output_folder = os.path.join(output_folder, 'contours')
    if not os.path.exists(contour_output_folder):
        os.makedirs(contour_output_folder)

    # 定义深绿色的HSV范围
    if lower_green is None:
        lower_green = np.array([20, 30, 30])  # 根据需要调整
    if upper_green is None:
        upper_green = np.array([100, 255, 255])  # 根据需要调整

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 获取文件的完整路径
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        contour_output_path = os.path.join(contour_output_folder, filename)

        # 检查文件是否为图像
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 读取图像
            image = cv2.imread(input_path)

            # 检查图像是否成功加载
            if image is not None:
                # 转换到HSV颜色空间
                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                # 创建深绿色的掩膜
                mask = cv2.inRange(hsv_image, lower_green, upper_green)

                # 使用掩膜去除深绿色部分
                result_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))

                # 去噪（使用高斯模糊）
                #denoised_image = cv2.GaussianBlur(result_image, (5, 5), 0)

                # 将图像转换为灰度图像
                gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

                # 直方图均衡
                equalized_image = cv2.equalizeHist(gray_image)

                # 形态学操作：腐蚀（用于去除小白点）
                erode_kernel = np.ones(erode_kernel_size, np.uint8)
                eroded_image = cv2.erode(equalized_image, erode_kernel, iterations=1)

                # 形态学操作：膨胀（用于连接近邻对象）
                dilate_kernel = np.ones(dilate_kernel_size, np.uint8)
                dilated_image = cv2.dilate(eroded_image, dilate_kernel, iterations=1)

                # 二值化
                _, binary_image = cv2.threshold(dilated_image, threshold_value, 255, cv2.THRESH_BINARY)

                # 形态学操作来闭合边缘
                kernel = np.ones(morph_kernel_size, np.uint8)
                closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

                # 保存处理后的二值图像
                cv2.imwrite(output_path, closed_image)
                print(f"Processed and saved binary image: {output_path}")

                # 提取轮廓
                contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 创建一个空白图像用于绘制轮廓
                contour_image = np.zeros_like(binary_image)

                # 轮廓近似
                for contour in contours:
                    # 根据轮廓的周长调整epsilon值以获得更好的拟合
                    epsilon = 0.001 * cv2.arcLength(contour, True)
                    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                    cv2.drawContours(contour_image, [approx_contour], -1, (255), 1)

                # 保存带有近似轮廓的图像
                cv2.imwrite(contour_output_path, contour_image)
                print(f"Processed and saved contour image: {contour_output_path}")
            else:
                print(f"Failed to load image: {input_path}")
        else:
            print(f"Skipped non-image file: {input_path}")


# 输入文件夹和输出文件夹的路径
input_folder = 'mango/downloadfiles1/archive/data/shixunfiles/3a2b8256a18499c8b84a46b04cc63d8c_1705368625397'  # 替换为你的输入文件夹路径
output_folder = 'output'

# 默认为127和(5,5)
threshold_value = 230
morph_kernel_size = (12, 12)  #边缘闭合效果
dilate_kernel_size = (3, 3)  # 膨胀操作的内核大小
erode_kernel_size = (7, 7)  # 腐蚀操作的内核大小

# 处理文件夹中的图像
preprocess_images(input_folder, output_folder, threshold_value, morph_kernel_size, dilate_kernel_size,
                  erode_kernel_size,
                  lower_green=np.array([35, 100, 100]), upper_green=np.array([85, 255, 255]))