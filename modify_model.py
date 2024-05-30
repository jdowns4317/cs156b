# models = ['densenet', 'resnet']
models = ['densenet']
num_epochs = ['2', '3', '5', '10']
# types = ['', 'drop', 'parallel']
types = ['drop']

for model in models:
    for ne in num_epochs:
        for t in types:
            with open(f'{model}{ne}final{t}.py', 'r') as f:
                base = f.readlines()
            base = [line.replace("df.dropna(inplace=True)", "df.dropna(subset=[feature], inplace=True)") for line in base]

            with open(f'{model}{ne}final{t}.py', 'w') as f:
                f.writelines(base)

print("done") 