import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

from advbench.lib import reporting, plotting

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot CVaR results')
    parser.add_argument('--input_dir', type=str, required=True)
    args = parser.parse_args()

    sns.set(style='darkgrid', font_scale=1.8, font='Palatino')

    results = reporting.load_records(args.input_dir, depth=1)
    losses = reporting.load_records(args.input_dir, results_fname='losses', depth=1)

    g = sns.FacetGrid(
        results, 
        col='Eval-Method', 
        hue='Train-Alg', 
        palette="colorblind",
        height=5,
        legend_out=True,
        col_wrap=4)
    
    g.map(plt.plot, 'Epoch', 'Accuracy', linewidth=3)
    g.set_titles(col_template='{col_name}')

    handles, labels = g.axes[0].get_legend_handles_labels()
    g.fig.legend(
        handles, 
        labels, 
        ncol=5, 
        bbox_to_anchor=(0.8,0.1),
        frameon=False)

    plt.subplots_adjust(bottom=0.2)
    plt.savefig('cvar_accuracies.png')
    plt.close()

    sns.set(font_scale=1.5, font='Palatino')

    g = sns.lineplot(
        data=losses,
        x='Epoch',
        y='Loss',
        hue='Train-Alg')
    g.set(ylabel='CVaR')

    plotting.adjust_legend_fontsize(g.axes, 15)

    plt.subplots_adjust(bottom=0.2)
    plt.savefig('cvar_losses.png')