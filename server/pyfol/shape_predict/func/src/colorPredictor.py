from sklearn.model_selection import train_test_split

from func import fed
import os.path
import pandas as pd
from tqdm import tqdm

dataset = pd.read_csv('../../Pillbox.csv')
dataset = dataset[dataset.has_image == True]
dataset = dataset[['ID', 'splimage', 'splcolor_text']]
dataset = dataset[dataset.splimage != 'no_product_image']
print(dataset)

train, test = train_test_split(dataset, test_size=0.3, random_state=1)
# print(train)
for index, row in tqdm(train.iterrows()):
    img = row.splimage + '.jpg'
    path_image = os.path.join('..', '..', 'pillbox_production_images_full_201812', img)
    prediction = fed.colorPrediction(path_image)
    train.loc[index, 'predict_color'] = prediction

dataset.to_csv('../../dataset_afterpredColor_train.csv')

# for index, row in tqdm(test.iterrows()):
#     img = row.splimage + '.jpg'
#     path_image = os.path.join('..', '..', 'pillbox_production_images_full_201812', img)
#     prediction = fed.colorPrediction(path_image)
#     test.loc[index, 'predict_color'] = 'prediction'
#
# dataset.to_csv('../../dataset_afterpredColor_test.csv')
