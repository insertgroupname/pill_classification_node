import pandas as pd
import os
from func import fed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('../Pillbox.csv')

# we only consider dataset that only have images.
dataset = dataset[dataset.has_image == True]
dataset = dataset[['ID', 'splimage', 'splshape_text', 'splcolor_text']]
dataset = dataset[dataset.splimage != 'no_product_image']
# For check with img file if it exist! one time usage

# for img in dataset['splimage']:
#     img = img + '.jpg'
#     path = os.path.join('..', 'pillbox_production_images_full_201812', img)
#     dataset['exist'] = os.path.exists(path)
#
# a = dataset[dataset.exist == True]
# a.to_csv('../fail_image.csv')

# import image for validating
# print(dataset.splcolor_text.value_counts()>10)
# a = dataset.splcolor_text.value_counts()
# a.to_csv('../color_count.txt',',')
cond = dataset.splshape_text == 'RECTANGLE'
dataset.loc[cond, 'splshape_text'] = 'QUADRANGLE'
cond2 = dataset.splshape_text == 'DIAMOND'
dataset.loc[cond2, 'splshape_text'] = 'QUADRANGLE'
cond3 = dataset.splshape_text == 'SQUARE'
dataset.loc[cond3, 'splshape_text'] = 'QUADRANGLE'
cond4 = dataset.splshape_text == 'TRAPEZOID'
dataset.loc[cond4, 'splshape_text'] = 'FREEFORM'
cond4 = dataset.splshape_text == 'HEXAGON (6 SIDED)'
dataset.loc[cond4, 'splshape_text'] = 'FREEFORM'
cond4 = dataset.splshape_text == 'OCTAGON (8 SIDED)'
dataset.loc[cond4, 'splshape_text'] = 'FREEFORM'
cond4 = dataset.splshape_text == 'PENTAGON (5 SIDED)'
dataset.loc[cond4, 'splshape_text'] = 'FREEFORM'
cond4 = dataset.splshape_text == 'TEAR'
dataset.loc[cond4, 'splshape_text'] = 'OVAL'
cond4 = dataset.splshape_text == 'DOUBLE CIRCLE'
dataset.loc[cond4, 'splshape_text'] = 'ROUND'
cond4 = dataset.splshape_text == 'BULLET'
dataset.loc[cond4, 'splshape_text'] = 'FREEFORM'
cond4 = dataset.splshape_text == 'SEMI-CIRCLE'
dataset.loc[cond4, 'splshape_text'] = 'FREEFORM'

cond4 = dataset.splshape_text == 'OVAL'
dataset.loc[cond4, 'splshape_text'] = 'CAPSULE_or_OVAL'
cond4 = dataset.splshape_text == 'CAPSULE'
dataset.loc[cond4, 'splshape_text'] = 'CAPSULE_or_OVAL'

#
# print(dataset)
# train,test= train_test_split(dataset,test_size=0.3, stratify=dataset['splshape_text'],random_state=1)

train,test= train_test_split(dataset,test_size=0.3,random_state=1)


for index,row in tqdm(train.iterrows()):
    img = row.splimage + '.jpg'
    path = os.path.join('..', 'pillbox_production_images_full_201812', img)
    number_polygon, shape = fed.shapeDetector(path)
    train.loc[index,'number_polygon'] = number_polygon
    train.loc[index,'predict_shape'] = shape

    # break

train.to_csv('../dataset_afterpred_train_shape.csv')


for index,row in tqdm(test.iterrows()):
    img = row.splimage + '.jpg'
    path = os.path.join('..', 'pillbox_production_images_full_201812', img)
    number_polygon, shape = fed.shapeDetector(path)
    test.loc[index,'number_polygon'] = number_polygon
    test.loc[index,'predict_shape'] = shape

    # prediction = fed.colorPrediction(path)
    # test.loc[index, 'predict_color'] = prediction

test.to_csv('../dataset_afterpred_test_shape.csv')