import json
import os
from PIL import Image, ImageDraw

def create_sub_dir(parent_dir,sub_dir_name):
    # 定义子路径的路径名称 Define the path to the subdirectory
    sub_dir_path = os.path.join(parent_dir, sub_dir_name)
    # Create the subdirectory if it doesn't exist
    if not os.path.exists(sub_dir_path):
        os.mkdir(sub_dir_path)
    return sub_dir_path


def draw_bboxes_coco_annotations(augmented_image_path_name, coco_annotations,generated_bbox_image_path):
    # Load the image using PIL
    image = Image.open(augmented_image_path_name)
    # Get the file name of the image
    file_name = image.filename.split('\\')[-1]
    new_filename = file_name.replace("augmented_", "")
    # Find the corresponding image ID based on the file name
    image_id = None
    for img in coco_annotations['images']:
        if img['file_name'] == new_filename:
            image_id = img['id']
            break

def draw_bboxes_augmented_annotations(augmented_path, augmented_image_path_name, augmented_annotations_path_name,generated_bbox_image_path):
    # 使用Pillow加载 经过图像增强的图片 Load the augmented image using PIL
    image = Image.open(augmented_image_path_name)

    # 加载其对应的注释信息 Load the corresponding annotation
    with open(augmented_path+"\\"+augmented_annotations_path_name, 'r') as f:
        augmented_annotations = json.load(f)
    # 利用Pillow库中的函数imagedraw创建一个画笔对象 Create a new image object for drawing
    draw = ImageDraw.Draw(image)

    # 在图像上画 被bbox Draw bounding boxes on the image
    outline_color = 0 # bbox的外边框颜色被设置为黑色 black color for the bounding boxes

    #遍历在augmented_annotations文件中所有的 子标注（每个bbox的坐标及长宽信息）
    for annotation in augmented_annotations:
        #读取标注中的bbox、category_id信息
        bbox = annotation['bbox']
        category_id = annotation['category_id']
        category = "arrow"  # 临时使用hardcode进行写入 Temporary category value
        #对bbox进行解包操作，读取x轴坐标（左上角）,y轴坐标（左上角）,w宽，h高
        x, y, w, h = bbox
        # 根据从该子标注中读取的信息，在该图像上画bbox。 Draw bbox on the image
        draw.rectangle([x, y, x + w, y + h], outline=outline_color, width=2)
        #在该图像上写入标注的文字描述
        draw.text((x, y - 10), category, fill=outline_color)

    # 保存本张已经画完bbox的增强图片至输出路径 Save the image with the bounding boxes
    image_name = os.path.basename(augmented_image_path_name).split('.')[0]  # 从json文件中提取出图像名称 Extract the image name from the JSON file name
    output_path = os.path.join(generated_bbox_image_path, f"{image_name}.jpg") #利用os库的join函数创建被除了完成的增强图像的输出路径名
    image.save(output_path)
