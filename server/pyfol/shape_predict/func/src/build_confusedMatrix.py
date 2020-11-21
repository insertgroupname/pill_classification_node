import pandas as pd
import sys
from func import confusionMatrix as cF
from sklearn.metrics import classification_report


def print_polygon(dataframe):
    print('Round')
    print(dataframe[dataframe['splshape_text'] == 'ROUND'].number_polygon.value_counts())
    print('CAPSULE_or_OVAL')
    print(dataframe[dataframe['splshape_text'] == 'CAPSULE_or_OVAL'].number_polygon.value_counts())
    print('TRIANGLE')
    print(dataframe[dataframe['splshape_text'] == 'TRIANGLE'].number_polygon.value_counts())
    print('QUADRANGLE')
    print(dataframe[dataframe['splshape_text'] == 'QUADRANGLE'].number_polygon.value_counts())
    print('FREEFORM')
    print(dataframe[dataframe['splshape_text'] == 'FREEFORM'].number_polygon.value_counts())



data = pd.read_csv('../../dataset_afterpred_test.csv')

# print(data.number_polygon.value_counts())
# print(data.predict_shape.value_counts())
# print(data.splshape_text.value_counts())
class_label = ['ROUND','CAPSULE_or_OVAL','TRIANGLE','QUADRANGLE','FREEFORM']
#
res = cF.generate_confusion_matrix(data,class_label)
cF.print_confusion_matrix(res, class_label, '../../Confused_Matrix_test.txt')
with open('../../Confused_Matrix_test2.txt', 'w') as f:
    sys.stdout = f
    a = classification_report(data['splshape_text'],data['predict_shape'],target_names=class_label)
    print(a)
    print_polygon(data)


