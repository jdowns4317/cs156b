# %%
import os
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#pid folder
main_folder = '../../../data/train'

# Define the transformation to convert images to tensors
transform = transforms.Compose([
    transforms.Resize((30, 30)),  # Resize the image if necessary
    transforms.ToTensor()           # Convert images to PyTorch tensors
])

# Function to load an image and convert it to a tensor
def load_image(image_path):
    with Image.open(image_path) as img:
        return transform(img)

# %%
df = pd.read_csv('../../../data/student_labels/train2023.csv')
outcomes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
dfs = {}
for col in outcomes:
    temp_df = df[['Path', col]].copy()
    temp_df = temp_df.dropna(subset = [col])
    tensors = []
    for path in list(temp_df['Path']):
        if path[0] != 't':
            continue # skip the incorrect filepath discovered in train2023.csv
        tensors.append(load_image(os.path.join('../../../data', path)))
    dfs[col] = (tensors, list(temp_df[col]))

# %%
test_df = pd.read_csv('../../../data/student_labels/test_ids.csv')
test_outcomes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
test_dfs = []

for path in list(test_df['Path']):
    test_dfs.append(load_image(os.path.join('../../../data', path)))

# %%
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(900, 20),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(20, 3)
)


# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# %%
def train_nn(data_loader):
    data, target = data_loader

    target = [int(n + 1) for n in target]

    data = torch.stack(data)
    target = torch.tensor(target)
    dataset = TensorDataset(data, target)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model.train()

    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Erase accumulated gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate loss
            loss = loss_fn(output, target)

            # Backward pass
            loss.backward()

            # Weight update
            optimizer.step()

    return model

# %%
def get_output(train_data, test_data):
    model = train_nn(train_data)

    test_data = torch.stack(test_data)
    test_dataset = TensorDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    model.eval()
    test_preds = torch.zeros(len(test_data), 1)
    start_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            data_tensor = batch[0]
            output = model(data_tensor)
            pred = output.argmax(dim=1, keepdim=True)
            end_idx = start_idx + pred.size(0)
            test_preds[start_idx: end_idx] = pred
            start_idx = end_idx
    return test_preds
            

# %%
features = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Pneumonia", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]

classification_dict = {}
classification_dict["Id"] = test_df['Id']

for feature in features:
    if len(dfs[feature][0]) == 0:
        classification_dict[feature] = [0 for i in range(len(test_dfs))]
    else:
        classification_dict[feature] = get_output(dfs[feature], test_dfs)
        classification_dict[feature] = [tensor.item() - 1 for tensor in classification_dict[feature]]

# print(classification_dict)
submission_df = pd.DataFrame(classification_dict)

# print(submission_df.shape)
submission_df.to_csv('gobeavers_submission.csv', index=False)


