import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt


def predict_(x):
    per = x[0]*100
    if per >= 50.0:
        return 'OVAL'
    else:
        return 'CAPSULE'

model = load_model('./server/pyfol/model_oval/')

import sys
test_url = 'public/img/capsule1.jpg'
# img = load_img('insertimgurl',target_size=(256,256))
img = load_img(test_url,target_size=(256,256))
plt.imshow(img)
plt.show()
img_array = img_to_array(img)*(1./255.)
img_array = tf.expand_dims(img_array, 0)
predictor = model.predict(img_array)
score = predictor[0]
print(
    'It has prediction percentage is %.49f, so the result would be %s' % (score*100, predict_(score))
)

