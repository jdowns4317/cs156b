import os

models = ['densenet', 'resnet']
num_epochs = ['2', '3', '5', '10']

for model in models:
    for ne in num_epochs:
        os.rename(f'{ne}{model}final.py', f'{model}{ne}final.py')
        os.rename(f'{ne}{model}finaldrop.py', f'{model}{ne}finaldrop.py')
        os.rename(f'{ne}{model}finalparallel.py', f'{model}{ne}finalparallel.py')

print("done")