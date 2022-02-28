import pandas as pd
import os, glob
import click
from pathlib import Path
import matplotlib.pyplot as plt

def plot(f, x, y):
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("A test graph")
    plt.plot(x,y,label = 'id %s'%f)
    plt.legend()
    plt.savefig(f, dpi=300, bbox_inches='tight', pad_inches = 0)


def get_files(path, dfname):
    files = []
    print("path:",path)
    print("dfname:",dfname)
    if not dfname:
        print("No file name provided")
    else:
        if len(dfname) > 1:
            files = list(dfname)
        else:
            dfname = dfname[0]
            print("No file name provided")
            tpath = os.path.join(path, dfname)
            if Path(tpath).is_file():
                files = [tpath]
            else:
                files = []
                for root, dirs, _files in os.walk(path):
                    for _file in _files:
                        root_file = os.path.join(root, _file)
                        if all(s in root_file for s in dfname.split("|")):
                            files.append(root_file)
    return files

@click.command()
@click.argument("fname", nargs=-1, type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--fid",
    "-fid",
    default="",
    type=str,
    help=""
)
def main(fname, path, fid):
   files = get_files(path, fname)
   dfs = []
   fff = fname[0]
   for f in files:
       df = pd.read_table(f,index_col=0, names=range(12))
       dft = (f, df)
       dfs.append(dft)
   dfs = sorted(dfs,  key = lambda x:x[0])
   print("after sorting")
   fn = f"/home/pouramini/heatmaps/{fid}/"
   Path(fn).mkdir(parents=True, exist_ok=True)
   x = range(len(dfs))
   df = dfs[0][1]
   params = list(df.index)
   #for param in params:
   for l in range(12):
       fig = plt.figure()  # create a figure
       ax = fig.add_subplot(111)
       ax.set_xlabel("samples")
       ax.set_ylabel(fname)
       ax.set_title(f"{fname}-{l}")
       for param in params:
       #for l in range(12):
           y = []
           for f, df in dfs:
               y.append(df.loc[param, l]) #.sum())
           ax.plot(x,y,label = '%s'%param)
       fn = f"/home/pouramini/heatmaps/{fid}/{fff}-{l}.png"
       print(fn)
       ax.legend()
       fig.savefig(fn, dpi=300, bbox_inches='tight', pad_inches = 0)

main()
