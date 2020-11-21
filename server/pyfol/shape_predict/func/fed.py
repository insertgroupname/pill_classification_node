import imutils
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from imutils import contours as ct
import numpy as np
import os
import sys
# sys.path.append("..")
import base64
from io import StringIO
from PIL import Image

# from src.color_recognition_api import color_histogram_feature_extraction, knn_classifier
from src.color_recognition_api import color_histogram_feature_extraction, knn_classifier

this_dir = "/home/chaichet_pderizer/work/pill/node/server/pyfol/shape_predict/func/"

def load_image(img):
    try:
        # img = cv2.imread(img)
        # img = img
        img2 = imutils.resize(img, width=360)
        return img2
    except Exception as e:
        print("Error : on load_img at line 24 ----agrument __"+str(e))


def show_image(img):
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    # plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
    plt.show()
    pass


def cvt2gray(img):
    try:
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e :
        print("Error : on cvt2gray line 38----agrument __"+str(e))

def create_mask(img, number=25, cont=3.5):
    # return cv2.threshold(img,number,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    try:
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, number, cont)
    except Exception as e:
        print("Error : on grid_shape_pred line 105----agrument __"+str(e))


def largest_contour(mask):
    try:
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # cv.CHAIN_APPROX_SIMPLE,CHAIN_APPROX_TC89_KCOS
        contours = imutils.grab_contours(contours)
        # where cnts is the variable in which contours are stored, replace it with your variable name
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        # sorts contours according to their area from largest to smallest.
        largestCont = contours[1]  # store the largest contour
        area = cv2.contourArea(largestCont)
        # print('area = ', area)
        return largestCont
    except Exception as e:
        print("Error : on largest_contour line 66----agrument __"+str(e))


def drawing_cont(img, contour, coefficient=0.037):
    try:
        epsilon = coefficient * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # print(len(approx))
        cv2.drawContours(img, approx, -1, (255, 0, 255), 5)
        return approx
    except Exception as e:
        print("Error : on drawing_cont line 77----agrument __"+str(e))



def shapePred(approx):
    try:
        a = 'FREEFORM'
        if len(approx) == 8:
            a = 'ROUND'
        # elif len(approx) == 7:
        #     a = 'OVAL'
        elif len(approx) == 6:
            a = 'CAPSULE_or_OVAL'
        elif len(approx) == 5:
            a = 'TRIANGLE'
        elif len(approx) == 4:
            a = 'QUADRANGLE'
        return a
    except Exception as e:
        print("Error : on grid_shape_pred line 105----agrument __"+str(e))

def Grid_shape_pred(path_img, block_size, constant, coefficient):
    try:
        img = load_image(path_img)
        gray = cvt2gray(img)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
        cont = largest_contour(mask)
        approx = drawing_cont(img, cont, coefficient)
        # show_image(img)
        pred = shapePred(approx)
        # cv2.imwrite(save_path,img)
        return len(approx), pred
    except Exception as e:
        print("Error : on grid_shape_pred line 105----agrument __"+str(e))


def shapeDetector(img):
    # img = load_image(path_img)
    try:
        gray = cvt2gray(img)
        mask = create_mask(gray, cont=4)
        # show_image(mask)
        cont = largest_contour(mask)
        # we use the second largest because normally largest are box of the picture
        approx = drawing_cont(img, cont)
        # show_image(img)
        pred = shapePred(approx)
        return len(approx), pred
    except Exception as e:
        print("Error : on shapeDetector line 139----agrument __"+str(e))


def roiImage(path_img):
    try:
        img = load_image(path_img)
        gray = cvt2gray(img)
        mask = create_mask(gray, cont=4)
        cont = largest_contour(mask)
        x, y, w, h = cv2.boundingRect(cont)
        ROI = img[y: y + h, x: x + w]
        croppedImg = ROI[22: -22, 22: -22]
        return croppedImg
    except Exception as e:
        print("roiImage ERROR : line 153 ---- "+str(e))


def colorPrediction(img):
    croppedImg = roiImage(img)
    new_path = this_dir+'src/color_recognition_api/'
    PATH = this_dir+'src/color_recognition_api/training.data'
    try:
        if (os.path.isfile(PATH) and os.access(PATH, os.R_OK)):
            pass
        else:
            open(PATH, 'w')
            color_histogram_feature_extraction.training()
        # get the prediction
        color_histogram_feature_extraction.color_histogram_of_test_image(croppedImg)
        prediction = knn_classifier.main(PATH, new_path+'test.data')
        return prediction
    except Exception as e:
        print("Error : on colorPrediction line 171------" +str(e))


def predict_oval(x):
    per = x[0] * 100
    if per >= 50.0:
        return 'OVAL'
    else:
        return 'CAPSULE'

def prep_img_b64():
    try:
        temp = sys.argv[1]
        temp = temp.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(temp), np.uint8)
        
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        print("prep img b64 ERROR : line 180 ---- "+str(e))

def main():
    # print(sys.argv)
    # 0 is file fed.py , 1 is arg image_path
    # path_img = sys.argv[1]
    img = prep_img_b64()
    model = load_model(this_dir+'ai_model')  # load tf model define path
    polygon, pred = shapeDetector(img)
    if pred == 'CAPSULE_or_OVAL':
        img = load_img(img, target_size=(256, 256))
        img_array = img_to_array(img) * (1. / 255.)
        img_array = tf.expand_dims(img_array, 0)
        predictor = model.predict(img_array)
        pred = predict_oval(predictor)
    print("Shape : "+str(pred))
    # pred is the prediction shape
    # for fluke color reimport and implement it below
    cropped_img = roiImage(img)
    prediction = colorPrediction(cropped_img)
    print("Color : "+str(prediction))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("MAIN ERROR : -----------" +str(e))