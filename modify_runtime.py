features = features = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
            "Lung Opacity", "Pneumonia", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices"]

models = ['densenet', 'resnet']
num_epochs = ['2', '3', '5', '10']
# types = ['', 'drop', 'parallel']
types = ['parallel']

for model in models:
    for ne in num_epochs:
        for t in types:
            for feature in features:
                new_model = f"{model}{ne}final{t}"

                with open(f'feature_{new_model}_{feature.lower().replace(" ", "_")}.sh', 'r') as f:
                    body = f.readlines()
                
                body = [line.replace("cpus-per-task=1", "cpus-per-task=5") for line in body]

                with open(f'feature_{new_model}_{feature.lower().replace(" ", "_")}.sh', 'w') as f:
                    f.writelines(body)

print("done")