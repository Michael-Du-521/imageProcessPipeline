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

# path parameters
labelmePath="C:\\Users\\ubei.DESKTOP-95T650K\\.conda\\envs\\imageProcessPipeline\\Scripts\\labelme.exe"
imagesFolderPath="D:\\Image_Process_Pipeline_06192023\\880防投反0518原图"
resizedImagesPath= "D:\\Image_Process_Pipeline_06192023\\880防投反0518原图-尺寸修改"
labelmeFolder=str(resizedImagesPath)
cocosFolder="D:\\Image_Process_Pipeline_06192023\\880防投反0518原图-尺寸修改-coco\\cocos"
augmentationFolder="D:\\Image_Process_Pipeline_06192023\\880防投反0518原图-尺寸修改-augmentation"
labelmeDir = str(resizedImagesPath)
generated_bbbox_image_path="D:\\Image_Process_Pipeline_06192023\\880防投反0518原图-尺寸修改-bbox"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # read images in specified folder
    imageFiles = glob.glob(imagesFolderPath + "\\*.jpg")
    print(imageFiles)
    # Loop over each image file, process it, and save the output image to the resized folder
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
    sub_dir_path_coco = labelMe.create_sub_dir(resizedImagesPath, sub_dir_name="cocos")
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
            deleteJsonFlag = True
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

    # Load the image and its corresponding bounding box annotations
    #sample_image_path="images/_Camera2_Kit2_NG_230517_143941.jpg"
    sample_image_resize_path = "D:\\Image_Process_Pipeline_06192023\\880防投反0518原图-尺寸修改"
    #"D:\Image_Process_Pipeline_06192023\880防投反0518原图-尺寸修改\_Camera2_Kit2_NG_230517_143941.jpg"
    sample_image_resize_augmented_path_name = "D:\\Image_Process_Pipeline_06192023\\880防投反0518原图-尺寸修改-augmentation\\augmented_image.jpg"
    image = np.array(Image.open(sample_image_resize_path+"\\_Camera2_Kit2_NG_230517_143941.jpg"))
    bboxes = coco_annotations['annotations'][0]['bbox']
    category_ids = coco_annotations['annotations'][0]['category_id']

    # import the image augmentation pipeline
    transform= imageAugmentation.transform0

    # Apply the augmentation to the image and its bounding box annotations
    augmented = transform(image=image, bboxes=[bboxes], category_id={category_ids})
    augmented_image =Image.fromarray(augmented['image'])
    #print(str(augmented))
    augmented_bboxes = augmented['bboxes']
    augmented_category_ids = augmented['category_id']

    # Save the augmented image and its corresponding bounding box annotations
    try:
        print("augmented image will be written into :",augmentationFolder)
        augmented_image_path=augmentationFolder+"\\augmentted_image.jpg"
        augmented_image.save(augmented_image_path)
        print("Image[{}] is saved successfully in :".format(augmented_image_path),augmentationFolder)
    except Exception as e:
        print("Error saving the image:", str(e))
    # Create a new Coco annotation dictionary
    coco_annotations_augmented= {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": {'bbox':[],'category_id':[]},
    "categories": [
  ]
}
    coco_annotations_augmented['annotations']['bbox'] = augmented_bboxes[0]
    coco_annotations_augmented['annotations']['category_id'] = augmented_category_ids[0]
    augmented_annotations_path=augmentationFolder+'\\augmented_annotations.json'
    with open(augmented_annotations_path, 'w') as f:
        json.dump(coco_annotations_augmented, f)

    #draw bbox and augmented image
    labelMe.draw_bboxes(sample_image_resize_augmented_path_name, coco_annotations_augmented, generated_bbbox_image_path)








