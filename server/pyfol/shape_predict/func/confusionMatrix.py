import pandas as pd
import sys


def generate_confusion_matrix(dataframe, class_labels):
    result = list()

    for i in range(len(class_labels)):
        result.append(list())
        for j in range(len(class_labels)):
            result[i].append(0)
    for index, row in dataframe.iterrows():
        label = row.splshape_text
        pred = row.predict_shape
        result[class_labels.index(label)][class_labels.index(pred)] += 1
    print('done')
    return result


def print_confusion_matrix(result, class_labels,file_path):
    with open(file_path, 'w') as f:
        sys.stdout = f
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                if result[i][j] == 0: continue
                print('Actual [{}], Predict [{}]: {}.'.format(class_labels[i], class_labels[j], result[i][j]))
