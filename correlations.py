import pandas as pd
import torch
import torch.multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt

mp.set_start_method('spawn', force=True)

# Check CUDA availability and select the appropriate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

features = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
            "Lung Opacity", "Pneumonia", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices"]

def correlation_testing(df, features):
    subset = df[features]
    corr_matrix = subset.corr()
    return corr_matrix

train_df = pd.read_csv('../../../data/student_labels/train2023_1.csv')
test_df = pd.read_csv('../../../data/student_labels/test_ids.csv')

corr_matrix = correlation_testing(train_df, features)
plt.figure(figsize=(15, 15))

# Drawing the heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'shrink': .8})

# Adding title and labels, adjust as needed
plt.title('Correlation Matrix of Features')
plt.xlabel('Features')
plt.ylabel('Features')

plt.savefig('corr_heatmap.png', dpi=300)

# Show the plot
plt.show()