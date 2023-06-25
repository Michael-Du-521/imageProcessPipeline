import random
import albumentations as A
import cv2

# Basic spatial-level transformations
hor_flip = A.HorizontalFlip(p=0.5)  # 水平翻转

ver_flip = A.VerticalFlip(p=0.5)  # 竖直翻转

angle = random.randint(0, 4) * 90 #random.randint(0, 3) generates a random integer between 0 and 4, inclusive.
rotate1 = A.Rotate(limit=[angle,angle],  p=1)  # 整90°旋转

rotate0 = A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=1)  # 旋转

rotate2=A.Rotate(limit=[(random.randint(0, 4) * 90),(random.randint(0, 4) * 90)],  p=1)


center_crop = A.CenterCrop(width=300, height=200, p=1)  # 中央裁剪

sharp = A.Sharpen(p=0.5) #锐化

random_brightness_contrast =A.RandomBrightnessContrast(p=0.2)# 随机亮度及对比度

# Declare an augmentation pipeline, p means probability
transformer0 = A.Compose([hor_flip,ver_flip,
   random_brightness_contrast],
    bbox_params=A.BboxParams(format='coco',label_fields=['category_id']))


