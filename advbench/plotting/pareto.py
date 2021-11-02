import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

if __name__ == '__main__':

    algs = ['PGD', 'FGSM', 'ALP', 'CLP', 'TRADES', 'MART',  'DALE']
    clean_accs = [83.8, 72.6, 75.9, 79.8, 80.7, 78.9, 86.7]
    adv_accs = [48.1, 40.7, 48.8, 48.4, 49.3, 49.9, 48.9]
    
    data = list(zip(algs, clean_accs, adv_accs))
    columns = ['Algorithm', 'Clean Accuracy', 'Adversarial Accuracy']
    df = pd.DataFrame(data, columns=columns)
    df['Clean error'] = 100 - df['Clean Accuracy']
    df['Adversarial error'] = 100 - df['Adversarial Accuracy']

    sns.set(style='darkgrid', font_scale=1.5, font='Palatino', palette='colorblind')

    g = sns.scatterplot(
        data=df,
        x='Clean error',
        y='Adversarial error',
        hue='Algorithm',
        marker='o',
        legend=False)
    g.set(
        title='Parteo Frontier')
        # xlim=(10, 30),  # clean
        # ylim=(40, 65))  # adversarial

    for i in range(df.shape[0]):
        plt.text(
            x=df['Clean error'][i] - 0.3, 
            y=df['Adversarial error'][i]+0.3, 
            s=df.Algorithm[i], 
            fontdict=dict(color='black', size=10))

    plt.subplots_adjust(bottom=0.15)
    plt.show()

