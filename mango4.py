import cv2
import numpy as np
import os


def preprocess_images(input_folder, output_folder, final_output_folder, threshold_value=127, morph_kernel_size=(5, 5),
                      dilate_kernel_size=(3, 3), erode_kernel_size=(3, 3), lower_green=None, upper_green=None,
                      min_area=500, max_aspect_ratio=2.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    contour_output_folder = os.path.join(output_folder, 'contours')
    ellipse_output_folder = os.path.join(output_folder, 'ellipses')
    if not os.path.exists(contour_output_folder):
        os.makedirs(contour_output_folder)
    if not os.path.exists(ellipse_output_folder):
        os.makedirs(ellipse_output_folder)

    if lower_green is None:
        lower_green = np.array([20, 30, 30])
    if upper_green is None:
        upper_green = np.array([100, 255, 255])

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        contour_output_path = os.path.join(contour_output_folder, filename)
        ellipse_output_path = os.path.join(ellipse_output_folder, filename)
        final_output_path = os.path.join(final_output_folder, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(input_path)
            if image is not None:
                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_image, lower_green, upper_green)
                result_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))

                gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
                equalized_image = cv2.equalizeHist(gray_image)

                erode_kernel = np.ones(erode_kernel_size, np.uint8)
                eroded_image = cv2.erode(equalized_image, erode_kernel, iterations=1)

                dilate_kernel = np.ones(dilate_kernel_size, np.uint8)
                dilated_image = cv2.dilate(eroded_image, dilate_kernel, iterations=1)

                _, binary_image = cv2.threshold(dilated_image, threshold_value, 255, cv2.THRESH_BINARY)

                kernel = np.ones(morph_kernel_size, np.uint8)
                closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

                cv2.imwrite(output_path, closed_image)
                print(f"Processed and saved binary image: {output_path}")

                contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                contour_image = np.zeros_like(binary_image)
                ellipse_image = np.zeros_like(image, dtype=np.uint8)  # Ensure this is uint8 for proper color blending

                for contour in contours:
                    epsilon = 0.01 * cv2.arcLength(contour, True)
                    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                    cv2.drawContours(contour_image, [approx_contour], -1, (255), 1)

                    if len(approx_contour) < 5:
                        print(
                            f"Warning: Skipping contour with {len(approx_contour)} points, not enough to fit ellipse.")
                        continue

                    # Fit ellipse to the contour
                    ellipse = cv2.fitEllipse(approx_contour)

                    # Calculate area and aspect ratio
                    area = cv2.contourArea(approx_contour)
                    (x, y), (MA, ma), angle = ellipse
                    aspect_ratio = MA / ma

                    if min_area <= area <= image.shape[0] * image.shape[1] and aspect_ratio <= max_aspect_ratio:
                        cv2.ellipse(ellipse_image, ellipse, (0, 255, 0), 2)

                # Overlay the ellipse image on the original image
                overlay_image = cv2.addWeighted(image, 0.8, ellipse_image, 0.2, 0)  # Adjust weights as needed

                cv2.imwrite(contour_output_path, contour_image)
                print(f"Processed and saved contour image: {contour_output_path}")

                cv2.imwrite(ellipse_output_path, ellipse_image)
                print(f"Processed and saved ellipse image: {ellipse_output_path}")

                cv2.imwrite(final_output_path, overlay_image)
                print(f"Processed and saved final image with ellipses: {final_output_path}")
            else:
                print(f"Failed to load image: {input_path}")
        else:
            print(f"Skipped non-image file: {input_path}")


input_folder = 'mango/downloadfiles1/archive/data/shixunfiles/3a2b8256a18499c8b84a46b04cc63d8c_1705368625397'  # 替换为你的输入文件夹路径
output_folder = 'output'
final_output_folder = 'final_output'

threshold_value = 230
morph_kernel_size = (12, 12)
dilate_kernel_size = (3, 3)
erode_kernel_size = (7, 7)

preprocess_images(input_folder, output_folder, final_output_folder, threshold_value, morph_kernel_size,
                  dilate_kernel_size,
                  erode_kernel_size, lower_green=np.array([35, 100, 100]), upper_green=np.array([85, 255, 255]),
                  min_area=500, max_aspect_ratio=2.0)