import os
from PIL import Image, ImageDraw

def create_sub_dir(parent_dir,sub_dir_name):
    # Define the path to the subdirectory
    sub_dir_path = os.path.join(parent_dir, sub_dir_name)
    # Create the subdirectory if it doesn't exist
    if not os.path.exists(sub_dir_path):
        os.mkdir(sub_dir_path)
    return sub_dir_path


def draw_bboxes(image_path, coco_annotation,generated_bbbox_image_path):
    # Load the image
    image = Image.open(image_path)

    # Draw each bounding box on the image
    bbox = coco_annotation['annotations']['bbox']
    category_id = coco_annotation['annotations']['category_id']
    #temperary placeholder
    category = "arrow"
    x, y, w, h = bbox
    # Create a drawing object
    draw = ImageDraw.Draw(image)

    #outline_color = (0, 255, 0)  # Green color
    outline_color =255

    # Draw bbox on the image
    draw.rectangle([x, y, x + w, y + h], outline=outline_color, width=2)
    draw.text((x, y - 10), category, fill=outline_color)

    # Save the image with bounding boxes
    image.save(generated_bbbox_image_path+"\\image.jpg")

