from pathlib import Path
import os, shutil
import click
from pytz import timezone
import datetime
import glob

tehran = timezone('Asia/Tehran')
now = datetime.datetime.now(tehran)
now = now.strftime('%Y-%m-%d-%H:%M')
@click.command()
@click.argument("fname", type=str)
@click.argument("dest_dir", type=str)
@click.option(
    "--path",
    "-p",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--move",
    "-m",
    is_flag=True,
    help=""
)
def mycopy(fname, path, move, dest_dir):
    delete_file = dest_dir == "dd"
    print_file = dest_dir == "pp"
    if glob.glob(fname):
        files = glob.glob(fname)
        files = [os.path.join(path, f) for f in files]
    else:
        files = []
        for root, dirs, _files in os.walk(path):
            for _file in _files:
                root_file = os.path.join(root, _file)
                if all(s in root_file for s in fname.split("+")):
                    files.append(root_file)
    if not dest_dir: dest_dir = now
    dest_dir = "/home/pouramini/share/" + dest_dir
    for file in files:
        if print_file:
            print("File: ", file)
            continue
        if delete_file:
            os.remove(file)
            print("Removing ", file)
            continue
        print("FROM:", file)
        rp = Path(file).relative_to(path)
        folders = Path(rp)
        parts = folders.parts[:-1]
        if parts:
            folder = os.path.join(dest_dir, "_".join(parts))
        else:
            folder = dest_dir

        print("TO:", folder)
        Path(folder).mkdir(parents=True, exist_ok=True)
        if not move:
            shutil.copy(file, folder)
        else:
            try:
                shutil.move(file, folder)
            except:
                print(file, " already exists!")

if __name__ == "__main__":
    mycopy()