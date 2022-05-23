from collections import defaultdict
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib.ticker as tkr
import click
import os
from pathlib import Path

num_layers = 12
def load(filename):
    result = []
    labels = []
    smallest = float('inf')
    largest = 0
    with open(filename) as file:
        for line in file:
            line = line.strip().split('\t')
            name_of_matrix = line[0]
            if "cossim" not in filename:
                amount_bigger = 2
                if "11b" in filename:
                    amount_bigger = 64
                if "large" in filename:
                    amount_bigger = 4
                if "small" in filename:
                    amount_bigger = 2
                if name_of_matrix == "wi" or name_of_matrix == "wo":
                    change = amount_bigger * 1.048 * 1000000
                else:
                    change = 1.048 * 1000000
            else:
                change = 1
            values = [float(x) / change for x in line[1:]]
            #if 'cossim' in filename:
            #     values = [1 - x for x in values]
            result.append(values)
            labels.append(name_of_matrix)
            for i in values:
                largest = max(i, largest)
                smallest = min(i, smallest)
    return np.asarray(result), labels
  
def get_s_l(filename, const):
    other_filename = filename.replace('encoder', 'decoder')
    if 'l1_decoder' in filename:
        other_filename = filename.replace('l1_decoder', 'l1_encoder')
    if 'cossim_decoder' in filename:
        other_filename = filename.replace('cossim_decoder', 'cossim_encoder')
    result = []
    labels = []
    smallest = float('inf')
    largest = 0
    with open(filename) as file:
        for line in file:
            line = line.strip().split('\t')
            name_of_matrix = line[0]
            if "cossim" not in filename:
                amount_bigger = 2
                if "11b" in filename:
                    amount_bigger = 64
                if "large" in filename:
                    amount_bigger = 4
                if "small" in filename:
                    amount_bigger = 2
                if name_of_matrix == "wi" or name_of_matrix == "wo":
                    change = amount_bigger * 1.048 * 1000000
                else:
                    change = 1.048 * 1000000
            else:
                change = 1
            values = [float(x) / change for x in line[1:]]
            # if 'cossim' in filename:
            #     values = [1 - x for x in values]
            result.append(values)
            labels.append(name_of_matrix)
            if name_of_matrix != "wo":
                for i in values:
                    largest = max(i, largest)
                    smallest = min(i, smallest)
    with open(other_filename) as file:
        for line in file:
            line = line.strip().split('\t')
            name_of_matrix = line[0]
            if "cossim" not in filename:
                amount_bigger = 2
                if "11b" in filename:
                    amount_bigger = 64
                if "large" in filename:
                    amount_bigger = 4
                if "small" in filename:
                    amount_bigger = 2
                if name_of_matrix == "wi" or name_of_matrix == "wo" and 'cossim' not in filename:
                    change = amount_bigger * 1.048 * 1000000
                else:
                    change = 1.048 * 1000000
            else:
                change = 1
            values = [float(x) / change for x in line[1:]]
            result.append(values)
            labels.append(name_of_matrix)
            if name_of_matrix != "wo":
                for i in values:
                    largest = max(i, largest)
                    smallest = min(i, smallest)
    if const == 0:
        return smallest, largest
    if const == 1:
        if 'cossim_decoder' in filename:
            return 0, 2.1e-3
        if 'cossim_encoder' in filename:
            return 0, 2.1e-3
        if 'l1_encoder' in filename:
            return 0, 2e-5
        if 'l1_decoder' in filename:
            return 0, 2e-5
    if const == 2:
        if 'cossim_decoder' in filename:
            return 0, 2.1e-3
        if 'cossim_encoder' in filename:
            return 0, 2.1e-3
        if 'l1_encoder' in filename:
            return 0, 1e-6
        if 'l1_decoder' in filename:
            return 0, 1e-6

@click.command()
@click.option(
    "--pattern",
    "-pat",
    default="htsv",
    type=str,
    help=""
)
@click.option(
    "--const",
    "-c",
    default=1,
    type=int,
    help=""
)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--folder",
    "-f",
    default="new",
    type=str,
    help=""
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help=""
)
def main(const, pattern, path,folder, force):
    #names_new = []
    #for n in names:
    #    names_new.append(n)
    #    names_new.append(n.replace('decoder', 'encoder'))
    #names = names_new

    #names_new = []
    #for n in names:
    #    names_new.append(n)
    #    names_new.append(n.replace('l1', 'cossim').replace('.tsv', '_new.tsv'))

    #names = names_new

    fpath = os.path.join(path, pattern)
    if Path(fpath).is_file():
        files = [fpath]
    else:
        files = []
        print("Searching ...", path)
        for root, dirs, _files in os.walk(path):
            for _file in _files:
                if all(s in _file for s in pattern.split("+")):
                    files.append(os.path.join(root, _file))
        #print("files: ",files)
    names = files
    Path(f"/home/pouramini/heatmaps/{folder}/").mkdir(parents=True, exist_ok=True)
    for j, name in enumerate(names):
        fname = Path(name).stem
        pname = name.replace("htsv","png")
        if Path(pname).is_file() and not force:
            print("already exists...")
            continue
        print(fname)
        matrix, labels = load(name)
        s, l = get_s_l(name, const)
        j = j // 2
        color = "Blues" if "l1_encoder" in name or "l1_decoder" in name else "Greens"
        new_labels = []
        for item in labels:
            if item == "wo":
                item = "w_o"
            if item == "wi":
                item = "w_i"
            if "x" in item:
                item = item[-1] + "_x"
            item = "$" + item + "$"
            new_labels.append(item)
        cbar_val = True
        formatter = tkr.ScalarFormatter()
        formatter.set_powerlimits((0, 0))
        if len(matrix.shape) < 2: 
            matrix = matrix.reshape(matrix.shape[-1],1)
            print(matrix.shape)
        myticks = [x for x in range(matrix.shape[1])] # if x % 3 == 0]
        #myticks = np.arange(min(myticks), max(myticks)+1, 3.0)
        ax = sns.heatmap(matrix, cmap=color, vmax=l, vmin=s, yticklabels=new_labels, xticklabels=myticks, square=True, cbar=cbar_val, cbar_kws={"shrink": .35, "format": formatter})
        cbar_axes = ax.figure.axes[-1]
        #ax.set_title(fname)
        #ax.set_xticks(ax.get_xticks()[::3])
        for i, label in enumerate(ax.get_xticklabels()):
            if i % 3 != 0:
                label.set_visible(False)

        cbar_axes.yaxis.label.set_size(2)
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        fig = ax.get_figure()
        for i in range(matrix.shape[0] + 1):
            ax.axhline(i, color='white', lw=2)
        for i in range(matrix.shape[1] + 1):
            ax.axvline(i, color='white', lw=2)
        fig.tight_layout()
        #np.savetxt(f"/home/pouramini/heatmaps/{folder}/{fname}.tsv", matrix, delimiter="\t")
        fig.savefig(pname, dpi=300, bbox_inches='tight', pad_inches = 0)
        fig.clf()


if __name__ == "__main__":
    main()
