features = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
            "Lung Opacity", "Pneumonia", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices"]

models = ['densenet', 'resnet']
num_epochs = ['2', '3', '5', '10']
types = ['', 'drop', 'parallel']
for model in models:
    for ne in num_epochs:
        for t in types:
            new_model = f"{model}{ne}final{t}"

            body = []
            for feature in features:
                body.append(f'sbatch feature_{new_model}_{feature.lower().replace(" ", "_")}.sh\n')

            with open(f"run_all_{new_model}.sh", 'w') as f:
                f.writelines(body)

print("done")