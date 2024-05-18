import pandas as pd
import numpy as np
import sys

model = sys.argv[1]

features = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Pneumonia', 'Pleural Effusion', 'Pleural Other',
            'Fracture', 'Support Devices']

# TODO change averages accordingly
averages = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Pleural Effusion',
            'Support Devices']

labels = pd.read_csv('../../../data/student_labels/train2023.csv')
test_df = pd.read_csv('../../../data/student_labels/test_ids.csv')
ids = list(test_df['Id'])

data = []

for feature in features:
    if feature in averages:
        mean = labels[feature].mean()
        data.append([mean] * len(ids))
    else:
        curr_df = pd.read_csv(f"results/{model}_{feature}.csv")
        data.append(curr_df[feature])

data = np.array(data, dtype=float)

submission = pd.DataFrame(data).transpose()
submission.columns = features
submission = submission.reindex(ids)
submission.index.name = 'Id'
submission.to_csv(f'results/{model}_joined.csv')