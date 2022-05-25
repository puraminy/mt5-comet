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
@click.option(
    "--show_dirs",
    "-d",
    is_flag=True,
    help=""
)
@click.option(
    "--only_name",
    "-on",
    is_flag=True,
    help=""
)
@click.option(
    "--anywhere",
    "-a",
    is_flag=True,
    help=""
)
def mycopy(fname, path, move, dest_dir, show_dirs, only_name, anywhere):
    delete_file = dest_dir == "dd"
    print_file = dest_dir == "pp"
    if glob.glob(fname):
        files = glob.glob(fname)
        files = [os.path.join(path, f) for f in files]
    else:
        files = []
        for root, dirs, _files in os.walk(path):
            if show_dirs:
                for _file in dirs:
                    root_file = os.path.join(root, _file)
                    if anywhere:
                        if all(s in root_file for s in fname.split("+")):
                            files.append(root_file)
                    else:
                        if all(s in _file for s in fname.split("+")):
                            files.append(root_file)
            else:
                for _file in _files:
                    root_file = os.path.join(root, _file)
                    if anywhere:
                        if all(s in root_file for s in fname.split("+")):
                            files.append(root_file)
                    else:
                        if all(s in _file for s in fname.split("+")):
                            files.append(root_file)
    if not dest_dir: dest_dir = now
    for ii, file in enumerate(files):
        if print_file:
            if show_dirs:
                if only_name:
                    print(ii, " cd ", Path(file).stem)
                else:
                    print(ii, " cd ", file)
            else:
                if only_name:
                    print(ii, " vi ", Path(file).stem)
                else:
                    print(ii, " vi ", file)
            continue
        if delete_file:
            if show_dirs:
                if Path(file).exists():
                    shutil.rmtree(file)
            else:
                os.remove(file)
            print("Removing ", file)
            continue
        print("FROM:", file)
        if not Path(dest_dir).exists():
            rp = Path(file).relative_to(path)
            folders = Path(rp)
            parts = folders.parts[:-1]
            if parts:
                folder = os.path.join(dest_dir, "_".join(parts))
            else:
                folder = dest_dir

            Path(folder).mkdir(parents=True, exist_ok=True)
        else:
            if ".." in dest_dir:
                folder = os.path.join(str(Path(file).parent), dest_dir)
                folder = Path(folder).resolve()
        if folder == Path(file).parent.resolve():
            print("Same directory")
            continue
        print("TO:", folder)
        if not move:
            shutil.copy(file, folder)
        else:
            try:
                shutil.move(file, folder)
            except:
                print(file, " already exists!")

if __name__ == "__main__":
    mycopy()
