import numpy as np
import argparse
import prettytable
import pandas as pd
import sys
import os

from advbench.lib import reporting, misc
from advbench import datasets

#TODO(AR): Currently no support for multiple trials

def scrape_results(df, trials, adv, split='Validation'):

    assert split in ['Validation', 'Test']

    all_dfs = []
    for trial in trials:
        trial_df = df[(df['Trial-Seed'] == trial) & (df['Eval-Method'] == adv) \
            & (df.Split == split)]

        # extract the row and epoch with the best performance for given adversary
        best_row = trial_df[trial_df.Accuracy == trial_df.Accuracy.max()]
        best_epoch = best_row.iloc[0]['Epoch']
        best_path = best_row.iloc[0]['Output-Dir']

        best_df = df[(df.Epoch == best_epoch) & (df['Output-Dir'] == best_path) \
            & (df['Trial-Seed'] == trial)]
        all_dfs.append(best_df)

    return pd.concat(all_dfs, ignore_index=True)

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(description='Collect results')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--depth', type=int, default=1, help='Results directories search depth')
    args = parser.parse_args()

    sys.stdout = misc.Tee(os.path.join(args.input_dir, 'results.txt'), 'w')

    records = reporting.load_records(args.input_dir, depth=args.depth)

    eval_methods = records['Eval-Method'].unique()
    dataset_names = records['Dataset'].unique()
    train_algs = records['Train-Alg'].unique()
    trials = records['Trial-Seed'].unique()
    
    for dataset in dataset_names:
        last_epoch = vars(datasets)[dataset].N_EPOCHS - 1

        for adv in eval_methods:

            # one table for each dataset/eval_method pair
            t = prettytable.PrettyTable()
            t.field_names = ['Training Algorithm', *[f'{m} Accuracy' for m in eval_methods], 'Output-Dir']
            print(f'\nSelection method: {adv} accuracy.')
            for alg in train_algs:
                df = records[(records['Dataset'] == dataset) & (records['Train-Alg'] == alg)]
                best_df = scrape_results(df, trials, adv, split='Test')
                test_df = best_df[best_df.Split == 'Test']

                accs = [test_df[test_df['Eval-Method'] == m].iloc[0]['Accuracy'] for m in eval_methods]
                output_dir = test_df.iloc[0]['Output-Dir']
                t.add_row([alg, *accs, output_dir])

            print(t)

                
                

