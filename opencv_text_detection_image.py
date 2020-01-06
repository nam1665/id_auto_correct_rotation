# USAGE
# python opencv_text_detection_image.py --image images/lebron_james.jpg
# --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import pytesseract
import re
import os

def auto_correct_rotation(img_path):
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str,
                    help="path to input image", default="./images/test1.jpg")
    ap.add_argument("-east", "--east", type=str,
                    help="path to input EAST text detector",default="./frozen_east_text_detection.pb")
    ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                    help="minimum probability required to inspect a region")
    ap.add_argument("-w", "--width", type=int, default=320,
                    help="resized image width (should be multiple of 32)")
    ap.add_argument("-e", "--height", type=int, default=320,
                    help="resized image height (should be multiple of 32)")
    args = vars(ap.parse_args())

    # load the input image and grab the image dimensions
    image = cv2.imread(img_path)
    orig = image.copy()
    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (args["width"], args["height"])
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    net = cv2.dnn.readNet(args["east"])

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    startx_value = []

    starty_value = []

    endx_value = []

    endy_value = []



    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < args["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            startx_value.append(startX)
            starty_value.append(startY)
            endx_value.append(endX)
            endy_value.append(endY)

            confidences.append(scoresData[x])


    if(startx_value):

        a1 = endx_value[0] - startx_value[0]
        b1 = endy_value[0] - starty_value[0]

        a2 = endx_value[1] - startx_value[1]
        b2 = endy_value[1] - starty_value[1]

        horizontal = ""
        vertical = ""

        if (((a1 - b1) > 0) and (a2 - b2) > 0):
            horizontal = True
            print("horizontal")
        else:
            vertical = True
            print("vertical")
        newdata = pytesseract.image_to_osd(img_path)
        orient = re.search('(?<=Rotate: )\d+', newdata).group(0)
        ori_confi = re.search('(?<=Orientation confidence: )\d+', newdata).group(0)
        id_img = cv2.imread(img_path)

        # get image height, width
        (h, w) = id_img.shape[:2]
        # calculate the center of the image
        center = (w / 2, h / 2)

        angle90 = 90
        angle180 = 180
        angle270 = 270

        scale = 1.0

        if (horizontal == True and int(orient) == 180):
            id_img = cv2.imread(img_path)
            img_rotate_180 = cv2.rotate(id_img, cv2.ROTATE_180)
            name = os.path.basename(img_path)
            cv2.imwrite('./images/' + name, img_rotate_180)
            print("rotated 180")

        elif (vertical == True and int(orient) == 90 and float(ori_confi) > 1):
            # id_img = cv2.imread(img_path)
            # img_rotate_90 = cv2.rotate(id_img, cv2.ROTATE_90)
            name = os.path.basename(img_path)
            # cv2.imwrite('./images/' + name, img_rotate_90)
            M = cv2.getRotationMatrix2D(center, angle90, scale)
            rotated90 = cv2.warpAffine(img_path, M, (h, w))
            cv2.imwrite('./images/' + name, rotated90)

            print("rotated 90")

        elif (vertical == True and int(orient) == 270 and float(ori_confi) > 1):
            id_img = cv2.imread(img_path)
            img_rotate_270 = cv2.rotate(id_img, cv2.ROTATE_270)
            name = os.path.basename(img_path)
            cv2.imwrite('./images/' + name, img_rotate_270)
            print("rotated 270")
        else:
            print("do nothing")

    else:
        return "cannot detect text-line"






