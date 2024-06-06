import pandas as pd
import numpy as np

features_dict = {'No Finding': '2densenetfinaldrop', 
                 'Enlarged Cardiomediastinum': '2resnetfinaldrop', 
                 'Cardiomegaly': '2densenetfinaldrop',
                 'Lung Opacity': '2resnetfinaldrop', 
                 'Pneumonia': '5resnetfinalparallel', 
                 'Pleural Effusion': '2densenetfinaldrop', 
                 'Pleural Other': 'average',
                 'Fracture': '10resnetfinaldrop',
                 'Support Devices': '2resnetfinaldrop'}

labels = pd.read_csv('../../../data/student_labels/train2023.csv')
test_df = pd.read_csv('../../../data/student_labels/test_ids.csv')
data_df = {}
data_df['Id'] = list(test_df['Id'])

for feature in features_dict.keys():
    if features_dict[feature] == 'average':
        mean = labels[feature].mean()
        data_df[feature] = [mean] * len(data_df['Id'])
    else:
        curr_df = pd.read_csv(f"results/{features_dict[feature]}{feature.replace(' ', '_')}.csv")
        data_df[feature] = curr_df[feature]

data = pd.DataFrame(data_df)

data.to_csv(f'results/best.csv', index=False)
print("done")