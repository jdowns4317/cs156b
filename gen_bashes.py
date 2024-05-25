features = features = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
            "Lung Opacity", "Pneumonia", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices"]

new_model = "10resnet18"

with open('feature_resnet_pneumonia.sh', 'r') as f:
    base = f.readlines()

for feature in features:
    body = base.copy()
    body[-1] = base[-1].replace("resnet", new_model)
    body[-1] = body[-1].replace("Pneumonia", feature)
    with open(f'feature_{new_model}_{feature.lower().replace(" ", "_")}.sh', 'w') as f:
        f.writelines(body)

print("done")