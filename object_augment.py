import albumentations as A
import cv2
import numpy as np

def augment_object(image):
    if image is None:
        raise ValueError("Input image is None")

    # Разделяем изображение и альфа-канал
    original_alpha = None
    if image.shape[2] == 4:
        b, g, r, original_alpha = cv2.split(image)
        rgb = cv2.merge([b, g, r])
    elif image.shape[2] == 3:
        rgb = image.copy()
        original_alpha = np.ones_like(rgb[:, :, 0]) * 255  # Создаем непрозрачный альфа-канал
    else:
        raise ValueError(f"Unsupported number of channels: {image.shape[2]}")

    # Преобразуем в RGB для Albumentations
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Определяем аугментации с учетом альфа-канала
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),  # Обновили параметр 'value' на 'limit'
        A.Affine(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.7, 
                 border_mode=cv2.BORDER_CONSTANT),  # Используем Affine вместо ShiftScaleRotate
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.1),
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.CLAHE(p=0.1),
        A.CoarseDropout(num_holes=8, min_height=8, min_width=8, p=0.5),  # Обновили параметры для CoarseDropout
        A.ElasticTransform(p=0.2),
        A.GridDistortion(p=0.2),
        A.OpticalDistortion(p=0.2),
    ], additional_targets={'mask': 'mask'})

    # Применяем аугментации
    augmented = transform(image=rgb, mask=original_alpha)
    aug_rgb = augmented['image']
    aug_alpha = augmented['mask']

    # Собираем обратно в BGRA
    aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
    return cv2.merge([aug_bgr[:, :, 0], aug_bgr[:, :, 1], aug_bgr[:, :, 2], aug_alpha])

def overlay_image(background, overlay, x=0, y=0):
    if overlay.shape[2] != 4:
        overlay = add_alpha_channel(overlay)
    
    h, w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]

    # Проверка выхода за границы
    if x >= bg_w or y >= bg_h:
        return background
    if x + w < 0 or y + h < 0:
        return background

    # Область наложения
    crop_x1 = max(0, -x)
    crop_y1 = max(0, -y)
    crop_x2 = min(w, bg_w - x)
    crop_y2 = min(h, bg_h - y)

    # Область в фоне
    bg_x1 = max(0, x)
    bg_y1 = max(0, y)
    bg_x2 = min(bg_w, x + w)
    bg_y2 = min(bg_h, y + h)

    # Вырезаем нужные части
    overlay_cropped = overlay[crop_y1:crop_y2, crop_x1:crop_x2]
    alpha = overlay_cropped[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha

    # Наложение с учетом альфа-канала
    for c in range(3):
        background[bg_y1:bg_y2, bg_x1:bg_x2, c] = (
            alpha * overlay_cropped[:, :, c] + 
            alpha_inv * background[bg_y1:bg_y2, bg_x1:bg_x2, c]
        )

    return background

def add_alpha_channel(image):
    if image.shape[2] == 3:
        alpha = np.ones_like(image[:, :, 0]) * 255
        return cv2.merge([image[:, :, 0], image[:, :, 1], image[:, :, 2], alpha])
    return image
