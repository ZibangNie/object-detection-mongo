import cv2
import numpy as np
import os


def load_and_preprocess_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.resize(img, (256, 256))  # Resize to 256x256 to standardize size
            images.append(img)
    return images


def locate_and_draw_mangoes(images):
    all_locations = []
    for image in images:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Adjusted HSV ranges for mango colors, assuming mangoes are bright greenish
        lower_mango = np.array([40, 100, 100])  # Lower bound for mango color (Hue, Saturation, Value)
        upper_mango = np.array([90, 255, 255])  # Upper bound for mango color
        mask = cv2.inRange(hsv_image, lower_mango, upper_mango)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        locations = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Adding a minimum area threshold to filter out noise
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 2.0:  # Aspect ratio for mango-like shapes
                    locations.append((x, y, w, h))
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Drawing rectangle
        all_locations.append(locations)
    return images, all_locations


def save_marked_images(images, all_locations, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, image in enumerate(images):
        for (x, y, w, h) in all_locations[i]:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_folder, f'marked_image_{i}.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


# Example usage
folder_path = r'mango/downloadfiles1/archive/data/shixunfiles/3a2b8256a18499c8b84a46b04cc63d8c_1705368625397'  # Set your folder path here
output_folder = 'multi_mango_2'  # Output folder for marked images
images = load_and_preprocess_images(folder_path)
marked_images, locations = locate_and_draw_mangoes(images)
save_marked_images(marked_images, locations, output_folder)