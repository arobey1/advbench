import matplotlib.pyplot as plt
import numpy as np

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

def show_bar_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.3f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.3f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)