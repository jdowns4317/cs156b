features = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
            "Lung Opacity", "Pneumonia", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices"]

new_model = "densenet"

for feature in features:
    with open(f'feature_{new_model}_{feature.lower().replace(" ", "_")}.sh', 'r') as f:
        body = f.readlines()
    body = [line.replace("30", "160") for line in body]
    with open(f'feature_{new_model}_{feature.lower().replace(" ", "_")}.sh', 'w') as f:
        f.writelines(body)