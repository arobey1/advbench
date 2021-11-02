import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

from advbench.lib import reporting, plotting

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot loss and accuracy')
    parser.add_argument('--input_dir', type=str, required=True)
    args = parser.parse_args()

    meters = reporting.load_records(args.input_dir, results_fname='meters', depth=0)
    results = reporting.load_records(args.input_dir, depth=0)

    loss_cols = [c for c in meters.columns if 'loss' in c]

    loss_meters = pd.melt(
        meters, 
        id_vars=['Epoch'], 
        value_vars=loss_cols,
        var_name='loss type',
        value_name='loss value')

    sns.set(style='darkgrid', font_scale=1.5, font='Palatino')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # plot training/test accuracies
    g = sns.lineplot(
        data=results, 
        x='Epoch', 
        y='Accuracy', 
        hue='Eval-Method',
        ax=ax1, 
        marker='o')
    name_dict = {'ERM': 'Clean', 'PGD_Linf': 'Adversarial'}
    g.set(title='Test accuracy')
    plotting.remove_legend_title(ax1, name_dict=name_dict)

    # plot training losses -- divided into clean and adv
    g = sns.lineplot(
        data=loss_meters, 
        x='Epoch', 
        y='loss value', 
        hue='loss type',
        ax=ax2,
        marker='o')
    g.set(ylabel='Loss', title='Training loss')    
    name_dict = {l: l.capitalize() for l in loss_cols}
    plotting.remove_legend_title(ax2, name_dict=name_dict)

    plt.subplots_adjust(bottom=0.15)

    save_path = os.path.join(args.input_dir, 'acc_and_loss.png')
    plt.savefig(save_path)