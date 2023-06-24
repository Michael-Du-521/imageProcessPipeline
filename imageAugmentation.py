import albumentations as A
import cv2

# Basic spatial-level transformations
hor_flip = A.HorizontalFlip(p=0.5)  # 水平翻转

ver_flip = A.VerticalFlip(p=1)  # 竖直翻转

rotate = A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=1)  # 旋转

crop = A.CenterCrop(width=300, height=200, p=1)  # 中央裁剪

sharp = A.Sharpen(p=0.5) #锐化

random_brightness_contrast =A.RandomBrightnessContrast(p=0.2)

# Declare an augmentation pipeline, p means probability
transform0 = A.Compose([
    hor_flip,
   random_brightness_contrast],
    bbox_params=A.BboxParams(format='coco',label_fields=['category_id']))


