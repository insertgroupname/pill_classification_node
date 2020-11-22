import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import base64
model = load_model('./server/pyfol/pizza_model/')

import sys
test_url = 'public/img/pizza-hut.jpg'

def predict_pizza(x):
    per = x[0]*100
    if per >= 75.0:
        return 'pizza'
    else:
        return 'not pizza'

def main():
    # img = load_img('insertimgurl',target_size=(256,256))
    try:
        if(sys.argv[1]!="test"):
            
            temp = sys.argv[1]
            ntemp = temp[(temp.index(",")):]
            i = base64.b64decode(ntemp)
            img = tf.io.decode_image(
                i, 
                channels=3,
                dtype=tf.uint8,
                expand_animations=False
            )
            img = tf.image.resize(img, [256,256])

        else:   
            img = load_img(test_url,target_size=(256,256))
        # plt.imshow(img)
        # plt.show()
        img_array = img_to_array(img)*(1./255.)
        img_array = tf.expand_dims(img_array, 0)
        predictor = model.predict(img_array)
        score = predictor[0]
        print('It has pizzaness %.22f'%(score*100) + '%' + ' which is %s' % (predict_pizza(score)))
        
    except Exception as e:
        print("\n"+str(e.message)+"\n"+str(e.args))
        
main()