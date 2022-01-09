import numpy as np
import argparse
import prettytable
import pandas as pd
import sys
import os
import itertools

from advbench.lib import reporting, misc
from advbench import datasets

def scrape_results(df, trials, loss_metric, split='Test'):

    all_best_dfs, all_last_dfs = [], []
    for trial in trials:
        trial_df = df[(df['Trial-Seed'] == trial) & (df['Eval-Method'] == loss_metric) \
            & (df.Split == split)] 

        best_row = trial_df[trial_df.Loss == trial_df.Loss.min()]
        best_epoch = best_row.iloc[0]['Epoch']
        best_path = best_row.iloc[0]['Output-Dir']

        last_row = df.iloc[-1]
        last_epoch = last_row['Epoch']
        last_path = last_row['Output-Dir']

        best_df = df[(df.Epoch == best_epoch) & (df['Output-Dir'] == best_path) \
            & (df['Trial-Seed'] == trial)]
        last_df = df[(df.Epoch == last_epoch) & (df['Output-Dir'] == last_path) \
            & (df['Trial-Seed'] == trial)]
        all_best_dfs.append(best_df)
        all_last_dfs.append(last_df)

    best_df = pd.concat(all_best_dfs, ignore_index=True)
    last_df = pd.concat(all_last_dfs, ignore_index=True)

    return best_df, last_df

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(description='Collect results')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--depth', type=int, default=1, help='Results directories search depth')
    args = parser.parse_args()

    sys.stdout = misc.Tee(os.path.join(args.input_dir, 'losses.txt'), 'w')

    records = reporting.load_records(args.input_dir, results_fname='losses', depth=args.depth)
    print(records)

    eval_methods = records['Eval-Method'].unique()
    dataset_names = records['Dataset'].unique()
    train_algs = records['Train-Alg'].unique()
    trials = records['Trial-Seed'].unique()

    for dataset in dataset_names:
        last_epoch = vars(datasets)[dataset].N_EPOCHS - 1

        for loss_metric in eval_methods:

            t = prettytable.PrettyTable()
            best_loss = [f'Best {m} Loss' for m in eval_methods]
            last_loss = [f'Last {m} Loss' for m in eval_methods]
            all_losses = list(itertools.chain(*zip(best_loss, last_loss)))
            t.field_names = ['Training Algorithm', *all_losses, 'Output-Dir']
            print(f'\nSelection method: {loss_metric} loss.')
            for alg in train_algs:
                df = records[(records['Dataset'] == dataset) & (records['Train-Alg'] == alg)]
                best_df, last_df = scrape_results(df, trials, loss_metric, split='Test')
                
                best_losses = [best_df[best_df['Eval-Method'] == m].iloc[0]['Loss'] for m in eval_methods]
                last_losses = [last_df[last_df['Eval-Method'] == m].iloc[0]['Loss'] for m in eval_methods]
                all_losses = list(itertools.chain(*zip(best_losses, last_losses)))
                output_dir = best_df.iloc[0]['Output-Dir']
                t.add_row([alg, *all_losses, output_dir])

            print(t)