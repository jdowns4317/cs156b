import os
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

# Check CUDA availability and select the appropriate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


features = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
            "Lung Opacity", "Pneumonia", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices"]

bs = 256
num_epochs = 3
w = 256
h = 256
nw = 4

class ImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, test=False):
        """
        Args:
            dataframe (DataFrame): DataFrame with image paths and optionally labels.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            test (bool, optional): If true, does not load labels.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx]['Path'])
        image = Image.open(img_name).convert('L')

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image.to(device)
        else:
            label = self.dataframe.iloc[idx]['Feature'] + 1
            label = torch.tensor(label, dtype=torch.long)
            return image.to(device), label.to(device)

# Transformation
transform = transforms.Compose([
    transforms.Resize((w, h)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Adjusted for one channel
])

# Load CSVs
print("DEBUG starting model")
train_df = pd.read_csv('../../../data/student_labels/train2023.csv')
test_df = pd.read_csv('../../../data/student_labels/test_ids.csv')

# def create_feature_df(df, feature):
#     df = df[df['Path'].str.startswith('t')]
#     df = df.dropna(subset=[feature]).query("Path.str.contains('frontal')")
#     df = df[['Path', feature]].reset_index(drop=True)
#     df.rename(columns={feature: 'Feature'}, inplace=True)
#     return df

def create_lateral_feature_df(df, feature):
    df = df[df['Path'].str.startswith('t')]
    df = df.dropna(subset=[feature]).query("Path.str.contains('lateral')")
    df = df[['Path', feature]].reset_index(drop=True)
    df.rename(columns={feature: 'Feature'}, inplace=True)
    return df

def create_frontal_feature_df(df, feature):
    df = df[df['Path'].str.startswith('t')]
    df = df.dropna(subset=[feature]).query("Path.str.contains('frontal')")
    df = df[['Path', feature]].reset_index(drop=True)
    df.rename(columns={feature: 'Feature'}, inplace=True)
    return df

dl_dict = {}
for feature in features:
    frontal_feature_df = create_frontal_feature_df(train_df, feature)
    lateral_feature_df = create_lateral_feature_df(train_df, feature)
    # feature_df = create_feature_df(train_df, feature)
    # train_dataset = ImageDataset(dataframe=feature_df, root_dir='../../../data', transform=transform)
    frontal_train_dataset = ImageDataset(dataframe=frontal_feature_df, root_dir='../../../data', transform=transform)
    lateral_train_dataset = ImageDataset(dataframe=lateral_feature_df, root_dir='../../../data', transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw)  # Adjust num_workers based on your system
    frontal_train_loader = DataLoader(frontal_train_dataset, batch_size=bs, shuffle=True, num_workers=nw)
    lateral_train_loader = DataLoader(lateral_train_dataset, batch_size=bs, shuffle=True, num_workers=nw)
    dl_dict[feature + "_frontal"] = frontal_train_loader
    dl_dict[feature + "_lateral"] = lateral_train_loader

print("DEBUG data loaded")

# Training Function
def train_nn(model, train_loader):
    # Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):  # Number of epochs
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()     # Zero the gradients
            output = model(data)      # Forward pass
            loss = loss_fn(output, target)  # Compute loss
            loss.backward()           # Backward pass
            optimizer.step()          # Update weights
    return model


# Function to get predictions
def get_output(train_loader, test_loader):
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # Convolutional layer
        nn.ReLU(),                                            # Activation layer
        nn.MaxPool2d(kernel_size=2, stride=2),                # Pooling layer
        nn.Dropout(0.5),                                      # Dropout layer
        
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),# Another convolutional layer
        nn.ReLU(),                                            # Activation layer
        nn.MaxPool2d(kernel_size=2, stride=2),                # Pooling layer
        nn.Dropout(0.5),                                      # Dropout layer
       
        nn.Flatten(),                                         # Flatten layer for transitioning to fully connected layer
        nn.Linear(64 * (w//4) * (h//4), 128),                 # Fully connected layer
        nn.ReLU(),                                            # Activation layer
        nn.Linear(128, 1)                         # Output layer
    )
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
    model.to(device)

    train_nn(model, train_loader)


    model.eval()
    test_preds = []
    test_probs = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_preds.extend(pred.flatten().tolist())
            
            probabilities = F.softmax(output, dim=1)  # converting logits to probabilities
            class_labels = torch.tensor([-1, 0, 1], device=probabilities.device)  # class labels tensor
            # Calculating expected values: sum of (probability * class_label) across each class
            expected_values = torch.sum(probabilities * class_labels, dim=1)  # shape [batch_size]
            test_probs.extend(expected_values.flatten().tolist())


    return test_preds, test_probs

def create_test_frontal_df(test_df):
    test_df = test_df[test_df['Path'].str.startswith('t')]
    test_df = test_df.query("Path.str.contains('frontal')")
    test_df = test_df.reset_index(drop=True)
    return test_df

def create_test_lateral_df(test_df):
    test_df = test_df[test_df['Path'].str.startswith('t')]
    test_df = test_df.query("Path.str.contains('lateral')")
    test_df = test_df.reset_index(drop=True)
    return test_df

test_dl_dict = {}
frontal_df = create_test_frontal_df(test_df)
lateral_df = create_test_lateral_df(test_df)
frontal_test_dataset = ImageDataset(dataframe=frontal_df, root_dir='../../../data', transform=transform)
lateral_test_dataset = ImageDataset(dataframe=lateral_df, root_dir='../../../data', transform=transform)
frontal_test_loader = DataLoader(frontal_test_dataset, batch_size=bs, shuffle=True, num_workers=nw)
lateral_test_loader = DataLoader(lateral_test_dataset, batch_size=bs, shuffle=True, num_workers=nw)
test_dl_dict["frontal"] = frontal_test_loader
test_dl_dict["lateral"] = lateral_test_loader

print("DEBUG test data loaded")

classification_dict = {}
classification_dict["Id"] = list(frontal_df['Id']) + list(lateral_df['Id'])

probs_dict = {}
probs_dict["Id"] = list(frontal_df['Id']) + list(lateral_df['Id'])

classification_dict_final = {}
probs_dict_final = {}
for feature in features:
    print(f"DEBUG running {feature}")
    classification_dict[feature + "_frontal"], probs_dict[feature + "_frontal"] = get_output(dl_dict[feature + "_frontal"], test_dl_dict["frontal"])
    classification_dict[feature + "_lateral"], probs_dict[feature + "_lateral"] = get_output(dl_dict[feature + "_lateral"], test_dl_dict["lateral"])
    classification_dict[feature + "_frontal"] = [pred - 1 for pred in classification_dict[feature + "_frontal"]]
    classification_dict[feature + "_lateral"] = [pred - 1 for pred in classification_dict[feature + "_lateral"]]
    classification_dict_final[feature] = list(classification_dict[feature + "_frontal"]) + list(classification_dict[feature + "_lateral"])
    probs_dict_final[feature] = list(probs_dict_final[feature + "_frontal"]) + list(probs_dict_final[feature + "_lateral"])

classification_dict_final["Id"] = classification_dict["Id"]
probs_dict_final["Id"] = probs_dict["Id"]

print("DEBUG exporting data")
submission_df = pd.DataFrame(classification_dict_final)
submission_df = submission_df.sort_values(by = "Id")
submission_df.to_csv('results/grey_submission.csv', index=False)

probs_df = pd.DataFrame(probs_dict_final)
probs_df = probs_df.sort_values(by = "Id")
probs_df.to_csv('results/grey_probs_submission.csv', index=False)



