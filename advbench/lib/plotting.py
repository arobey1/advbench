import matplotlib.pyplot as plt

def remove_legend_title(ax, name_dict=None):
    handles, labels = ax.get_legend_handles_labels()
    if name_dict is not None:
        labels = [name_dict[x] for x in labels]
    ax.legend(handles=handles, labels=labels)