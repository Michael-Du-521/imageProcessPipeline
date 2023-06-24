import albumentations as A
import cv2

# Basic spatial-level transformations
hor_flip = A.HorizontalFlip(p=1)  # 水平翻转

ver_flip = A.VerticalFlip(p=1)  # 竖直翻转

rotate = A.Rotate(p=1)  # 旋转

crop = A.CenterCrop(width=300, height=200, p=1)  # 中央裁剪

sharp = A.Sharpen(p=0.5) #锐化

# Declare an augmentation pipeline, p means probability
transform0 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2)],
    bbox_params=A.BboxParams(format='coco',label_fields=['category_id']))

transform1 = A.Compose([
    crop,
    sharp
])

transform2 = A.Compose([
    crop
])

