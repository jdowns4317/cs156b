features = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
            "Lung Opacity", "Pneumonia", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices"]

model = "10resnet18"

body = []
for feature in features:
    body.append(f'sbatch feature_{model}_{feature.lower().replace(" ", "_")}.sh\n')

with open(f"run_all_{model}.sh", 'w') as f:
    f.writelines(body)

print("done")