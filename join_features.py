import pandas as pd
import numpy as np

model = "resnet_probs"

features = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Pneumonia', 'Pleural Effusion', 'Pleural Other',
            'Fracture', 'Support Devices']

# TODO change averages accordingly
averages = ['No Finding']

labels = pd.read_csv('../../../data/student_labels/train2023.csv')
test_df = pd.read_csv('../../../data/student_labels/test_ids.csv')
data_df = {}
data_df['Id'] = list(test_df['Id'])


for feature in features:
    if feature in averages:
        mean = labels[feature].mean()
        data_df[feature] = [mean] * len(data_df['Id'])
    else:
        curr_df = pd.read_csv(f"results/{model}_{feature.replace(' ', '_')}.csv")
        data_df[feature] = curr_df[feature]

data = pd.DataFrame(data_df)

data.to_csv(f'results/joined_{model}.csv', index=False)
print("done")