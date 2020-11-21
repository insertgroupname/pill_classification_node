import os
import cv2
import numpy as np

this_path = "/home/chaichet_pderizer/work/pill/node/server/pyfol/shape_predict/func/src/color_recognition_api/"
def color_histogram_of_test_image(test_src_image):
    # load the image
    chans = cv2.split(test_src_image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue
            # print(feature_data)

    with open(str(this_path)+'test.data', 'w') as myfile:
        myfile.write(feature_data)


def color_histogram_of_training_image(img_name):
    # detect image color by using image file name to label training data
    if 'red' in img_name:
        data_source = 'RED'
    elif 'yellow' in img_name:
        data_source = 'YELLOW'
    elif 'green' in img_name:
        data_source = 'GREEN'
    elif 'orange' in img_name:
        data_source = 'ORANGE'
    elif 'white' in img_name:
        data_source = 'WHITE'
    elif 'black' in img_name:
        data_source = 'BLACK'
    elif 'blue' in img_name:
        data_source = 'BLUE'
    elif 'purple' in img_name:
        data_source = 'PURPLE'
    elif 'brown' in img_name:
        data_source = 'BROWN'
    elif 'pink' in img_name:
        data_source = 'PINK'
    # load the image
    image = cv2.imread(img_name)
    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    with open(str(this_path)+'training.data', 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')


def training():
    color = ['red', 'yellow', 'green', 'orange', 'white', 'black', 'blue', 'brown', 'pink', 'purple']
    for c in color:
        # Please check your own path if you got error from color_path
        # Using os.getcwd() for check the real path
        # color_path = os.path.join('..', 'color_training_dataset', c) for fluke
        color_path = os.path.join(str(this_path)+'color_training_dataset', c)
        # print(color_path)
        for file in os.listdir(color_path):
            file_path = os.path.join(color_path, file)
            # print(file_path)
            color_histogram_of_training_image(file_path)
