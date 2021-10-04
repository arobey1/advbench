import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os

from advbench.lib import reporting


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot learning curve')
    parser.add_argument('--input_dir', type=str, required=True)
    args = parser.parse_args()

    records = reporting.load_records(args.input_dir, depth=0)

    sns.set(style='darkgrid', font_scale=1.5)
    g = sns.relplot(data=records, x='Epoch', y='Accuracy', hue='Split', 
        col='Eval-Method', kind='line', marker='o')
    
    save_path = os.path.join(args.input_dir, 'learning_curve.png')
    plt.savefig(save_path)