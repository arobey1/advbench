import matplotlib.pyplot as plt
try:
    import wandb
except ImportError:
    print("Wandb not found, plots will not be logged.")

def remove_legend_title(ax, name_dict=None, fontsize=16):
    handles, labels = ax.get_legend_handles_labels()
    if name_dict is not None:
        labels = [name_dict[x] for x in labels]
    ax.legend(handles=handles, labels=labels, fontsize=fontsize)

def adjust_legend_fontsize(ax, fontsize):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, fontsize=fontsize)

def multicol_legend(ax, ncol=2):
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend.remove()
    ax.legend(handles, labels, ncol=ncol, loc='best')

def tick_density(plot, every=2, mod_val=1, axis='x'):
    ticks = plot.get_yticklabels() if axis == 'y' else plot.get_xticklabels()
    for ind, label in enumerate(ticks):
        if ind % every == mod_val:
            label.set_visible(True)
        else:
            label.set_visible(False)

def plot_perturbed_wandb(deltas, metric, name="loss", wandb_args = {}):
    data = [[x, y] for (x, y) in zip(deltas.tolist(), metric.tolist())]
    table = wandb.Table(data=data, columns = ["delta", name])
    wandb_dict = {"name" : wandb.plot.scatter(table, "delta", name)}
    wandb_dict.update(wandb_args)
    wandb.log(wandb_dict)