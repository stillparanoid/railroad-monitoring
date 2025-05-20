import os
import pandas as pd
import logging

import config
import cv2
import numpy as np
  
from object_augment import augment_object

log = logging.getLogger(__name__)

# Пути к папкам с данными
EXTRACTED_FRAMES_FOLDER = os.path.join(config.DATA_FOLDER, "extracted_frames")
PREPARED_OBJECTS_FOLDER = os.path.join(config.DATA_FOLDER, "prepared_objects")
OUTPUT_FOLDER = os.path.join(config.DATA_FOLDER, "synthetic_images")
 
# Загружаем категории с масштабами и приводим названия к нижнему регистру с заменой пробелов на _
CATEGORIES = pd.read_csv(os.path.join(config.DATA_FOLDER, "categories.csv"))
CATEGORIES["Object name"] = CATEGORIES["Object name"].str.replace(" ", "_").str.lower()

def main():
    # Собираем список всех кадров
    extracted_frames = []
    for root, _, files in os.walk(EXTRACTED_FRAMES_FOLDER):
        for file in files:
            extracted_frames.append(os.path.join(root, file))

    # Собираем список всех подготовленных объектов по категориям
    prepared_objects = {}
    for dir_name in os.listdir(PREPARED_OBJECTS_FOLDER):
        dir_path = os.path.join(PREPARED_OBJECTS_FOLDER, dir_name)
        if not os.path.isdir(dir_path):
            continue
        prepared_objects[dir_name] = []
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                prepared_objects[dir_name].append(file_path)

    # Создаем папку для синтетических изображений, если её нет
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Проходим по каждой категории и объектам, накладываем объекты на случайные кадры
    for category, objects in prepared_objects.items():
        for obj in objects:
            for frame in np.random.choice(extracted_frames, min(10, len(extracted_frames)), replace=False):
                try:
                    synthetic_image = combine_images(frame, obj, category)
                except Exception as e:
                    log.error(f"Error combining images: {e}")
                    continue

                output_filename = f"{os.path.splitext(os.path.basename(frame))[0]}_{category}_{os.path.splitext(os.path.basename(obj))[0]}.png"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                cv2.imwrite(output_path, synthetic_image)
                print(f"Saved synthetic image to {output_path}")

def combine_images(frame_path: str, object_path: str, category: str):
    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Failed to load frame image from {frame_path}")

    obj = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)
    if obj is None:
        raise FileNotFoundError(f"Failed to load object image from {object_path}")

    # Аугментируем объект (например, повороты, сдвиги и т.п.)
    obj = augment_object(obj)

    # Подгоняем размер объекта под фон, не увеличивая
    obj = resize_image(obj, frame.shape)

    # Масштабируем объект согласно фактору из CSV
    obj = rescale_image(obj, category)

    # Получаем случайные координаты для размещения объекта на фоне
    x, y = get_random_position(frame, obj)

    # Накладываем объект с прозрачностью
    synthetic_image = overlay_image(frame, obj, x, y)
    return synthetic_image

def resize_image(image, target_shape):
    target_height, target_width = target_shape[:2]
    height, width = image.shape[:2]

    scale = min(target_width / width, target_height / height, 1.0)  # Запрет увеличения
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def rescale_image(image, category):
    scale_factor_arr = CATEGORIES.loc[CATEGORIES["Object name"] == category, "scale factor"].values
    if len(scale_factor_arr) == 0:
        print(f"No scale factor found for category '{category}'. Using scale 1.0.")
        scale_factor = 1.0
    else:
        scale_factor = scale_factor_arr[0]

    if scale_factor <= 0:
        print(f"Invalid scale factor '{scale_factor}' for category '{category}'. Using scale 1.0.")
        scale_factor = 1.0

    print(f"Scaling category '{category}' with factor {scale_factor}")

    return cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

def get_random_position(background, overlay):
    bg_height, bg_width = background.shape[:2]
    overlay_height, overlay_width = overlay.shape[:2]

    if overlay_height > bg_height or overlay_width > bg_width:
        raise ValueError("Overlay image is larger than the background.")

    target_x = (bg_width - overlay_width) // 2
    target_y = (bg_height - overlay_height) * 3 // 4

    x = np.random.randint(target_x - target_x // 15, target_x + target_x // 15 + 1)
    y = np.random.randint(target_y - target_y // 7, target_y + target_y // 7 + 1)

    return x, y

def overlay_image(background, overlay, x=0, y=0):
    if overlay.shape[2] != 4:
        raise ValueError("Overlay image must have an alpha channel (4 channels).")

    overlay_height, overlay_width = overlay.shape[:2]

    alpha_mask = overlay[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha_mask
    overlay_rgb = overlay[:, :, :3]

    bg_height, bg_width = background.shape[:2]

    y1, y2 = max(y, 0), min(y + overlay_height, bg_height)
    x1, x2 = max(x, 0), min(x + overlay_width, bg_width)

    if y1 >= y2 or x1 >= x2:
        print("Overlay position is outside the background image. Skipping overlay.")
        return background

    overlay_y1 = max(0, -y)
    overlay_y2 = overlay_height - max(0, y + overlay_height - bg_height)
    overlay_x1 = max(0, -x)
    overlay_x2 = overlay_width - max(0, x + overlay_width - bg_width)

    roi = background[y1:y2, x1:x2].astype(float)
    overlay_region = overlay_rgb[overlay_y1:overlay_y2, overlay_x1:overlay_x2].astype(float)
    alpha = alpha_mask[overlay_y1:overlay_y2, overlay_x1:overlay_x2, np.newaxis]
    alpha_inv_region = alpha_inv[overlay_y1:overlay_y2, overlay_x1:overlay_x2, np.newaxis]

    blended = (alpha * overlay_region) + (alpha_inv_region * roi)
    blended = blended.astype("uint8")

    background[y1:y2, x1:x2] = blended

    return background

if __name__ == "__main__":
    main()
