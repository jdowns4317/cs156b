# models = ['densenet', 'resnet']
models = ['densenet']
num_epochs = ['2', '3', '5', '10']
types = ['', 'drop', 'parallel']

for model in models:
    for ne in num_epochs:
        for t in types:
            with open(f'{model}{ne}final{t}.py', 'r') as f:
                base = f.readlines()
            base = [line.replace("model.classifier = torch.nn.Linear(2208, 1)", "num_features = model.classifier.in_features\n\tmodel.classifier = torch.nn.Linear(num_features, 1)") for line in base]

            with open(f'{model}{ne}final{t}.py', 'w') as f:
                f.writelines(base)

print("done") 