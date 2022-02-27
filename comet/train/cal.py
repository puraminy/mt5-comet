import pandas as pd
import os, glob

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
   for f in files:
       print(f)
       df = pd.read_table(f,index_col=0, names=range(12))
       dft = (f, df)
       dfs.add(dft)
   dfs = sorted(dfs,  key = lambda x:x[0])
   print("after sorting")
   for f, df in dfs:
       print(f)
