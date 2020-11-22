import imutils
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from imutils import contours as ct
import os
import sys
cwd = str(os.path.split(__file__)[0])+'/'
img_path = os.getcwd()+'/public/img/uploads/';
# from src.color_recognition_api import color_histogram_feature_extraction, knn_classifier
from src.color_recognition_api import color_histogram_feature_extraction, knn_classifier


def load_image(path_img):
    img = cv2.imread(path_img)
    img2 = imutils.resize(img, width=360)
    return img2


def show_image(img):
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    # plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
    plt.show()
    pass


def cvt2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def create_mask(img, number=25, cont=3.5):
    # return cv2.threshold(img,number,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, number, cont)


def largest_contour(mask):
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


def drawing_cont(img, contour, coefficient=0.037):
    epsilon = coefficient * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    # print(len(approx))
    cv2.drawContours(img, approx, -1, (255, 0, 255), 5)
    return approx



def shapeDetector(path_img):
    img = load_image(path_img)
    gray = cvt2gray(img)
    mask = create_mask(gray, 0)
    # show_image(mask)
    cont = largest_contour(mask)
    # we use the second largest because normally largest are box of the picture
    approx = drawing_cont(img, cont)
    # show_image(img)
    pred = shapePred(approx)
    # cv2.imwrite(save_path,img)
    return len(approx), pred


def shapePred(approx):
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


def Grid_shape_pred(path_img, block_size, constant, coefficient):
    img = load_image(path_img)
    gray = cvt2gray(img)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
    cont = largest_contour(mask)
    approx = drawing_cont(img, cont, coefficient)
    # show_image(img)
    pred = shapePred(approx)
    # cv2.imwrite(save_path,img)
    return len(approx), pred


def shapeDetector(path_img):
    img = load_image(path_img)
    gray = cvt2gray(img)
    mask = create_mask(gray, cont=4)
    # show_image(mask)
    cont = largest_contour(mask)
    # we use the second largest because normally largest are box of the picture
    approx = drawing_cont(img, cont)
    # show_image(img)
    pred = shapePred(approx)
    return len(approx), pred


def roiImage(path_img):
    img = load_image(path_img)
    gray = cvt2gray(img)
    mask = create_mask(gray, cont=4)
    cont = largest_contour(mask)
    x, y, w, h = cv2.boundingRect(cont)
    ROI = img[y: y + h, x: x + w]
    croppedImg = ROI[22: -22, 22: -22]
    return croppedImg


def colorPrediction(path_img):
    croppedImg = roiImage(path_img)
    PATH = cwd+'src/color_recognition_api/training.data'
    PATH_test = cwd+'src/color_recognition_api/test.data'
    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        # print('training data is ready, classifier is loading...')
        pass
    else:
        open(PATH, 'w')
        color_histogram_feature_extraction.training()
    # get the prediction
    color_histogram_feature_extraction.color_histogram_of_test_image(croppedImg)
    prediction = knn_classifier.main(PATH, PATH_test)
    return prediction


def predict_oval(x):
    per = x[0] * 100
    if per >= 50.0:
        return 'OVAL'
    else:
        return 'CAPSULE'


def main():
    # print(sys.argv)
    # 0 is file fed.py , 1 is arg image_path
    path_img = sys.argv[1]
    model = load_model(cwd+'ai_model')  # load tf model define path
    polygon, pred = shapeDetector(img_path+path_img)
    if pred == 'CAPSULE_or_OVAL':
        img = load_img(img_path+path_img, target_size=(256, 256))
        img_array = img_to_array(img) * (1. / 255.)
        img_array = tf.expand_dims(img_array, 0)
        predictor = model.predict(img_array)
        pred = predict_oval(predictor)
    print(pred)
    # pred is the prediction shape
    # for fluke color reimport and implement it below
    # print("------------------")
    cropped_img = roiImage(img_path+path_img)
    # print("------------------")
    prediction = colorPrediction(img_path+path_img)
    print(prediction)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR BOI : ----------" + str(e))
