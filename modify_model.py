models = ['densenet', 'resnet']
num_epochs = ['2', '3', '5', '10']
types = ['', 'drop', 'parallel']

for model in models:
    for ne in num_epochs:
        for t in types:
            with open(f'{model}{ne}final{t}.py', 'r') as f:
                base = f.readlines()
            # base = [line.replace('', 'DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)') for line in base]
            body = []
            prev = None
            for line in base:
                if prev = None:
                    if "np.transpose" in line:
                        prev = line
                    else:
                        body.append(line)
                if prev != None:
                    body.append(line)
                    

            with open(f'{model}{ne}final{t}.py', 'w') as f:
                f.writelines(base)

print("done")