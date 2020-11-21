import pandas as pd
import sys
import numpy as np
from sklearn.metrics import classification_report

data = pd.read_csv('../../dataset_afterpred.csv')
class_label = np.append(data['splcolor_text'].unique(), 'NONE')
# res = cF.generate_confusion_matrix(data, class_label)
# cF.print_confusion_matrix(res, class_label, '../../Confused_Matrix2.txt')
with open('../../Confused_Matrix.txt', 'w') as f:
    sys.stdout = f
    a = classification_report(data['splcolor_text'], data['predict_color'], target_names=class_label)
    print(a)
