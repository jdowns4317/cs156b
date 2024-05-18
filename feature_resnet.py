import os
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys

features = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
            "Lung Opacity", "Pneumonia", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices"]

feature = sys.argv[1]

bs = 64
num_epochs = 3
w = 256
h = 256
nw = 4
torch.backends.cudnn.benchmark = True

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
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = self.dataframe.iloc[idx]['Feature'] + 1
            label = torch.tensor(label, dtype=torch.long)
            return image, label

# Transformation
transform = transforms.Compose([
    transforms.Resize(256),                # Resize the image to 256x256 pixels
    transforms.CenterCrop(224),            # Crop the image to 224x224 pixels
    transforms.ToTensor(),                 # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the image
                         std=[0.229, 0.224, 0.225])
])

# Load CSVs
print("DEBUG starting model")
train_df = pd.read_csv('../../../data/student_labels/train2023.csv')
test_df = pd.read_csv('../../../data/student_labels/test_ids.csv')

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
frontal_feature_df = create_frontal_feature_df(train_df, feature)
lateral_feature_df = create_lateral_feature_df(train_df, feature)
frontal_train_dataset = ImageDataset(dataframe=frontal_feature_df, root_dir='../../../data', transform=transform)
lateral_train_dataset = ImageDataset(dataframe=lateral_feature_df, root_dir='../../../data', transform=transform)
frontal_train_loader = DataLoader(frontal_train_dataset, batch_size=bs, shuffle=True)
lateral_train_loader = DataLoader(lateral_train_dataset, batch_size=bs, shuffle=True)
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
        print(f"DEBUG epoch: {epoch}")
        for data, target in train_loader:
            data, target = data, target
            optimizer.zero_grad()     # Zero the gradients
            output = model(data)      # Forward pass
            loss = loss_fn(output, target)  # Compute loss
            loss.backward()           # Backward pass
            optimizer.step()          # Update weights
    return model


# Function to get predictions
def get_output(train_loader, test_loader):
    model = models.densenet121(weights='DEFAULT')
    num_features = model.classifier.in_features  # Get the number of inputs for the existing layer
    model.classifier = torch.nn.Linear(num_features, 3)  # Replace with a new layer with 3 outputs

    model

    train_nn(model, train_loader)


    model.eval()
    test_preds = []
    test_probs = []
    with torch.no_grad():
        for data in test_loader:
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
frontal_test_dataset = ImageDataset(dataframe=frontal_df, root_dir='../../../data', transform=transform, test=True)
lateral_test_dataset = ImageDataset(dataframe=lateral_df, root_dir='../../../data', transform=transform, test=True)
frontal_test_loader = DataLoader(frontal_test_dataset, batch_size=bs, shuffle=True)
lateral_test_loader = DataLoader(lateral_test_dataset, batch_size=bs, shuffle=True)
test_dl_dict["frontal"] = frontal_test_loader
test_dl_dict["lateral"] = lateral_test_loader

print("DEBUG test data loaded")

classification_dict = {}
classification_dict["Id"] = list(frontal_df['Id']) + list(lateral_df['Id'])

probs_dict = {}
probs_dict["Id"] = list(frontal_df['Id']) + list(lateral_df['Id'])

classification_dict_final = {}
probs_dict_final = {}
print(f"DEBUG running {feature}")
classification_dict[feature + "_frontal"], probs_dict[feature + "_frontal"] = get_output(dl_dict[feature + "_frontal"], test_dl_dict["frontal"])
print(f"DEBUG got frontal")
classification_dict[feature + "_lateral"], probs_dict[feature + "_lateral"] = get_output(dl_dict[feature + "_lateral"], test_dl_dict["lateral"])
print(f"DEBUG got lateral")
classification_dict[feature + "_frontal"] = [pred - 1 for pred in classification_dict[feature + "_frontal"]]
classification_dict[feature + "_lateral"] = [pred - 1 for pred in classification_dict[feature + "_lateral"]]
classification_dict_final[feature] = list(classification_dict[feature + "_frontal"]) + list(classification_dict[feature + "_lateral"])
probs_dict_final[feature] = list(probs_dict[feature + "_frontal"]) + list(probs_dict[feature + "_lateral"])

classification_dict_final["Id"] = classification_dict["Id"]
probs_dict_final["Id"] = probs_dict["Id"]
    

print(f"DEBUG {feature} exporting data")
submission_df = pd.DataFrame(classification_dict_final)
submission_df = submission_df.sort_values(by = "Id")

feature_under = feature.replace(" ", "_")
submission_df.to_csv(f'results/dense_{feature_under}.csv', index=False)
probs_df = pd.DataFrame(probs_dict_final)
probs_df = probs_df.sort_values(by = "Id")
probs_df.to_csv(f'results/dense_probs_{feature_under}.csv', index=False)

