features = features = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
            "Lung Opacity", "Pneumonia", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices"]

models = ['densenet', 'resnet']
num_epochs = ['2', '3', '5', '10']
types = ['', 'drop', 'parallel']
for model in models:
    for ne in num_epochs:
        for t in types:
            new_model = f"{model}{ne}final{t}"

            with open('feature_resnet_pneumonia.sh', 'r') as f:
                base = f.readlines()

            for feature in features:
                body = base.copy()
                body[-1] = base[-1].replace("feature_resnet", new_model)
                body[-1] = body[-1].replace("Pneumonia", feature)
                with open(f'feature_{new_model}_{feature.lower().replace(" ", "_")}.sh', 'w') as f:
                    f.writelines(body)

print("done")