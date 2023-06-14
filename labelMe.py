import os

import cv2


def create_sub_dir(parent_dir,sub_dir_name):
    # Define the path to the subdirectory
    sub_dir_path = os.path.join(parent_dir, sub_dir_name)
    # Create the subdirectory if it doesn't exist
    if not os.path.exists(sub_dir_path):
        os.mkdir(sub_dir_path)
    return sub_dir_path


def draw_bboxes(image_path, coco_annotations):
    # Load the image
    image = cv2.imread(image_path)

    # Draw each bounding box on the image
    for annotation in coco_annotations['annotations']:
        bbox = annotation['bbox']
        category_id = annotation['category_id']
        category = coco_annotations['categories'][category_id]['name'] #in coco the categories also start at 0
        x, y, w, h = bbox
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(image, category, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Bboxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
