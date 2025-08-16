import cv2
import numpy as np
import os


def preprocess_and_extract_edges(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 获取文件的完整路径
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 检查文件是否为图像
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 读取图像
            image = cv2.imread(input_path)

            # 检查图像是否成功加载
            if image is not None:
                # 将图像转换为灰度图像
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # 对灰度图像进行CLAHE运算
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                clahe_image = clahe.apply(gray_image)

                # 使用Canny边缘检测器提取边缘
                # 这里选择了两个阈值，较低的阈值用于检测弱边缘，较高的阈值用于检测强边缘
                # 这两个阈值之间的边缘会被认为是强边缘的一部分，从而有助于连接弱边缘
                edges = cv2.Canny(clahe_image, threshold1=100, threshold2=200)

                # 边缘检测后的图像已经是二值的（黑白），但为了确保是纯黑白（0和255），可以进行如下处理
                # 实际上，Canny边缘检测已经返回了二值图像，这一步是多余的，但为了清晰起见，我还是保留了它
                _, binary_edges = cv2.threshold(edges, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

                # 保存处理后的图像
                cv2.imwrite(output_path, binary_edges)
                print(f"Processed and saved edge image: {output_path}")
            else:
                print(f"Failed to load image: {input_path}")
        else:
            print(f"Skipped non-image file: {input_path}")


# 输入文件夹和输出文件夹的路径
input_folder = 'mango/downloadfiles1/archive/data/shixunfiles/3a2b8256a18499c8b84a46b04cc63d8c_1705368625397'
output_folder = 'output'

# 处理文件夹中的图像
preprocess_and_extract_edges(input_folder, output_folder)
