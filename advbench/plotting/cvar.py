import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

from advbench.lib import reporting, plotting



if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Plot loss and accuracy')
    # parser.add_argument('--input_dir', type=str, required=True)
    # args = parser.parse_args()

    sns.set(style='darkgrid', font_scale=1.5, font='Palatino')
    plt.figure(figsize=(12,5))

    algs = [
        r'CVaR SGD ($\beta=0.05$)',
        r'CVaR SGD ($\beta=0.01$)',
        r'CVaR SGD ($\beta=0.1$)',
        'ERM',
        'ERM w/ Data Aug',
        'FGSM',
        'PGD'
    ]
    cvars = [0.298, 0.882, 0.259, 5.313, 0.997, 1.092, 0.357]
    data = list(zip(algs, cvars))
    df = pd.DataFrame(data, columns=['Algorithm', r'CVaR ($\beta=0.05$)'])

    g = sns.barplot(
        data=df,
        y='Algorithm',
        x=r'CVaR ($\beta=0.05$)',
    )
    plt.subplots_adjust(left=0.25, bottom=0.15)
    g.set(ylabel='')
    show_values(g, orient='h')
    plt.savefig('cvar.png')