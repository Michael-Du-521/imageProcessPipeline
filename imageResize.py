# code for image resize
import os
import cv2
import numpy as np
from PIL import Image


def image_resize(imagePath, resizeScale= np.zeros(2),inputImageSize=1024):
    # Check if the input image file exists and is readable
    if not os.path.exists(imagePath) or not os.access(imagePath, os.R_OK):
        raise ValueError("Input image file does not exist or is not readable: {}".format(imagePath))

    #imageMat = cv2.imread(imagePath)
    with Image.open(imagePath) as img:
        imageMat=np.array(img)
    # Check if the imageMat variable is None
    if imageMat is None:
        raise ValueError("Failed to read input image file: {}".format(imagePath))

    # Check if the input image is already the correct size
    if imageMat.shape[0] == inputImageSize and imageMat.shape[1] == inputImageSize:
        resizeScale[0] = resizeScale[1] = 1
        return imageMat


    if (imageMat.shape[1] > imageMat.shape[0]):
        resizeScale[0] = resizeScale[1] = imageMat.shape[1] / inputImageSize
        imageMat=cv2.resize(imageMat,(imageMat.shape[1],imageMat.shape[1]))
        outResizeMat = cv2.resize(imageMat,(1024,1024))
        return outResizeMat

    elif (imageMat.shape[1] == imageMat.shape[0]):
        resizeScale[0] = resizeScale[1] = imageMat.shape[1]/ inputImageSize;
        outResizeMat = cv2.resize(imageMat,(1024,1024))
        return outResizeMat;

    else:
        resizeScale[0]=resizeScale[1]=imageMat.shape[0]/inputImageSize
        newMat = cv2.transpose(imageMat)
        newMat=cv2.resize(newMat,(newMat.shape[1],newMat.shape[0]))
        newImageMat = cv2.transpose(newMat)
        outResizeMat = cv2.resize(newImageMat,(1024,1024))
        return outResizeMat

def draw_mat(outResizeMat):

    # Print the width and height of the outResizeMat matrix
    print("Width of Mat",outResizeMat.shape[1])
    print("Height of Mat",outResizeMat.shape[0])

    # Display the output image
    cv2.imshow("Output Image", outResizeMat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
