#依赖类库导入
import glob
import json
import os
import shutil
import subprocess

from PIL import Image
import labelme2coco
import imageResize
import labelMe
import imageAugmentation
import numpy as np
import albumentations as A

# 路径变量 path parameters
#原图文件夹路径
imagesFolderPath="D:\\Image_Process_Pipeline_06192023\\880防投反0518原图"
#labelme 可执行文件 路径
labelmePath="C:\\Users\\ubei.DESKTOP-95T650K\\.conda\\envs\\imageProcessPipeline\\Scripts\\labelme.exe"
resizedImagesPath= "D:\\Image_Process_Pipeline_06192023\\880防投反0518原图-尺寸修改"
labelmeFolder=str(resizedImagesPath)
cocosFolder="D:\\Image_Process_Pipeline_06192023\\880防投反0518原图-尺寸修改-coco\\cocos"
augmentationFolder="D:\\Image_Process_Pipeline_06192023\\880防投反0518原图-尺寸修改-augmentation"
labelmeDir = str(resizedImagesPath)
generated_bbox_image_path="D:\\Image_Process_Pipeline_06192023\\880防投反0518原图-尺寸修改-bbox"


# 点击左侧绿色按钮运行该脚本 Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # read images in specified folder
    imageFiles = glob.glob(imagesFolderPath + "\\*.jpg")
    # 逐个读取图像文件、处理它，并将其保存至 修改尺寸后的文件夹 Loop over each image file, process it, and save the output image to the resized folder
    for imagePath in imageFiles:
        print("Reading : ",imagePath)
        # Process the image file
        outResizeMat = imageResize.image_resize(imagePath)
        # Convert the NumPy array to a Pillow image object
        resizedImage = Image.fromarray(outResizeMat)
        # Save the output image to the resized folder
        resizedImagePath = resizedImagesPath +"\\"+ imagePath.split("\\")[-1]
        print("Writing : ",resizedImagePath)
        resizedImage.save(resizedImagePath)
    print("Successfully Resized to {}".format(resizedImagesPath))

    # Construct the command to open Labelme with the resized image as input
    command = '{} {}'.format(labelmePath, resizedImagesPath)
    # Use the subprocess module to open Labelme
    subprocess.call(command, shell=True)
    sub_dir_path_json = labelMe.create_sub_dir(resizedImagesPath,sub_dir_name="annotations")
    # Loop over all the files in the mixed directory
    for file_name in os.listdir(resizedImagesPath):
        # Check if the file is a JSON file
        if file_name.endswith(".json"):
            # Define the path to the original file
            file_path = os.path.join(resizedImagesPath, file_name)
            # Define the path to the new file location in the subdirectory
            new_file_path = os.path.join(sub_dir_path_json, file_name)
            # Move the file to the subdirectory
            shutil.copy(file_path, new_file_path)
            # Delete the origional jsons or not
            deleteJsonFlag = False
            if(deleteJsonFlag):
                # Delete the original file
                print("Delete [{}] in parent directory : {}".format(file_name,deleteJsonFlag))
                os.remove(file_path)
            else:
                print("Keep [{}] in parent directory. ".format(file_name))

    # read jsons in specified folder
    jsonFiles = glob.glob(sub_dir_path_json + "\\*.json")
    for jsonPath in jsonFiles:
        print("Reading : ",jsonPath)
    # set convert train split rate
    train_split_rate = 1
    # Convert the Labelme annotations to COCO format
    labelme2coco.convert(labelme_folder=labelmeFolder, export_dir=cocosFolder)

    #image augmentation process using albumentation library

    # Load the Coco format annotation file
    with open(str(cocosFolder)+"\\dataset.json",'r') as f:
        coco_annotations = json.load(f)

    # import the image augmentation pipeline
    transformer = imageAugmentation.transformer0

    # Iterate over the images in the dataset
    for image_data in coco_annotations['images']:
        image_id = image_data['id']
        image_filename = image_data['file_name']

        # Read the resized image
        image_path = os.path.join(resizedImagesPath, image_filename)
        image = Image.open(image_path)
        image_np = np.array(image)

        # Find the corresponding annotations for the image
        annotations = [annotation for annotation in coco_annotations['annotations'] if annotation['image_id'] == image_id]
        # Apply the augmentation to the image and its bounding box annotations
        augmented = transformer(image=image_np,bboxes=[ann['bbox'] for ann in annotations], category_id=[ann['category_id'] for ann in annotations])
        # Access the augmented image and annotations
        augmented_image_np = augmented['image']
        augmented_bboxes = augmented['bboxes']
        # Update the annotations with the transformed bounding boxes
        for i, ann in enumerate(annotations):
            ann['bbox'] = augmented_bboxes[i] # Update the annotations as needed
        augmented_annotations =annotations
        # Convert the augmented image back to PIL image
        augmented_image =Image.fromarray(augmented_image_np)


        # Save the augmented images corresponding bounding box annotations
        augmented_annotations_filename = f"augmented{image_filename.replace('.jpg', '.json')}"
        augmented_annotations_path = os.path.join(augmentationFolder, augmented_annotations_filename)
        with open(augmented_annotations_path, 'w') as f:
            json.dump(augmented_annotations, f)

        # Save the augmented image with modified filename
        augmented_image_filename = f"augmented{image_filename}"
        augmented_image_path_name = os.path.join(augmentationFolder, augmented_image_filename)
        augmented_image.save(augmented_image_path_name)
        #draw bbox and the augmented image at the same time
        labelMe.draw_bboxes_augmented_annotations(augmentationFolder,augmented_image_path_name,augmented_annotations_filename , generated_bbox_image_path)
print("\n"+"ImageProcessPipeline Ends Successfully!","Please change to directory "+generated_bbox_image_path+" see the final results, have a good day")
# Execute system command to open the folder
subprocess.Popen(f'explorer "{generated_bbox_image_path}"')








