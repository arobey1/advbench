import numpy as np
import argparse
import prettytable
import pandas as pd
import sys
import os

from advbench.lib import reporting, misc
from advbench import model_selection

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(description='Collect results')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--depth', type=int, default=1, help='Results directories search depth')
    parser.add_argument('--selection_methods', type=str, nargs='+', default=['LastStep', 'EarlyStop'])
    args = parser.parse_args()

    sys.stdout = misc.Tee(os.path.join(args.input_dir, 'results.txt'), 'w')

    selection_methods = [
        vars(model_selection)[s] for s in args.selection_methods
    ]

    train_args = misc.read_dict(
        os.path.join(args.input_dir, 'args.json')
    )
    selection_df = reporting.load_sweep_dataframes(
        path=args.input_dir,
        depth=1
    )
    selection_metrics = [
        k for k in selection_df.columns.values.tolist()
        if any(e in k for e in train_args['evaluators'])
    ]

    df = pd.melt(
        frame=selection_df,
        id_vars=['Split', 'Algorithm', 'trial_seed', 'seed', 'path', 'Epoch']
    ).rename(columns={'variable': 'Metric-Name', 'value': 'Metric-Value'})

    for method in selection_methods:
        for metric_name, metric_df in df.groupby('Metric-Name'):
            t = prettytable.PrettyTable()
            t.field_names = ['Algorithm', metric_name, 'Selection Method']

            for algorithm, algorithm_df in metric_df.groupby('Algorithm'):
                selection = method(algorithm_df)
                vals = selection.trial_values
                mean, sd = np.mean(vals), np.std(vals)
                t.add_row([
                    algorithm, f'{mean:.4f} +/- {sd:.4f}', method.NAME
                ])
            print(t)
