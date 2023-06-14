import glob
import os
import shutil
import subprocess
from PIL import Image
import labelme2coco
from pycocotools.coco import COCO
import imageResize
import labelMe

# path parameters
labelmePath="C:\\Users\\ubei.DESKTOP-95T650K\\.conda\\envs\\imageProcessPipeline\\Scripts\\labelme.exe"
imagesFolderPath="C:\\Users\\ubei.DESKTOP-95T650K\\Desktop\\880防投反0518原图"
resizedImagesPath= "C:\\Users\\ubei.DESKTOP-95T650K\\Desktop\\880防投反0518原图-尺寸修改"
labelmeFolder=str(resizedImagesPath)
cocosFolder="C:\\Users\\ubei.DESKTOP-95T650K\\Desktop\\880防投反0518原图-尺寸修改-coco\\cocos"
labelmeDir = str(resizedImagesPath)


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
            # Delete the orginal jsons or not
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
    train_split_rate = 0.85
    # Convert the Labelme annotations to COCO format
    labelme2coco.convert(labelme_folder=labelmeFolder, export_dir=cocosFolder)



