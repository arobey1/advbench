import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json
import os

from advbench.lib import reporting, plotting

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot primal dual')
    parser.add_argument('--input_dir', type=str, required=True)
    args = parser.parse_args()

    meters = reporting.load_records(args.input_dir, results_fname='meters', depth=0)
    results = reporting.load_records(args.input_dir, depth=0)

    with open(os.path.join(args.input_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)

    loss_meters = pd.melt(
        meters, 
        id_vars=['Epoch'], 
        value_vars=['clean loss', 'robust loss'],
        var_name='loss type',
        value_name='loss value')

    sns.set(style='darkgrid', font_scale=1.8, font='Palatino')
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))

    # plot training/test accuracies
    g = sns.lineplot(
        data=results, 
        x='Epoch', 
        y='Accuracy', 
        hue='Eval-Method',
        ax=ax1, 
        # marker='o',
        linewidth=4,
        palette=['#f26b22', '#6ed2fa', '#ff4d82'])
    name_dict = {'ERM': 'Clean', 'PGD_Linf': r'PGD$^{20}$', 'FGSM_Linf': 'FGSM'}
    g.set(title='Test accuracy', ylabel='Accuracy (%)')
    plotting.remove_legend_title(ax1, name_dict=name_dict, fontsize=16)
    

    # plot training losses -- divided into clean and adv
    g = sns.lineplot(
        data=loss_meters, 
        x='Epoch', 
        y='loss value', 
        hue='loss type',
        ax=ax2,
        # marker='o',
        linewidth=4,
        palette=['#6ed2fa', '#f26b22'])
    g.set(ylabel='Loss', title='Training loss')    
    ax2.axhline(
        hparams['g_dale_pd_margin'], 
        ls='--', 
        c='red',
        linewidth=2,
        zorder=10,
        label=r'Margin $\rho$')
    name_dict = {'clean loss': r'Nominal $\ell_{nom}$', 'robust loss': r'Robust $\ell_{ro}$', r'Margin $\rho$': r'Margin $\rho$'}
    plotting.remove_legend_title(ax2, name_dict=name_dict, fontsize=16)
    # plotting.tick_density(ax2, every=2, mod_val=1, axis='y')

    # plot dual variable
    g = sns.lineplot(
        data=meters,
        x='Epoch',
        y='dual variable',
        ax=ax3,
        # marker='o',
        linewidth=4,
        color='#f26b22')
    g.set(ylabel=r'Magnitude of $\nu$', title='Dual variable')

    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()

    save_path = os.path.join(args.input_dir, 'primal_dual.png')
    plt.savefig(save_path)
