
import os
from pathlib import Path

def find_files(fname, fpath, ftype="tsv"):
    if not fname:
        print("No file name provided")
    else:
        if type[fname] == list and len(fname) > 1:
            files = list(fname)
        else:
            fname = fname[0]
            path = os.path.join(fpath, fname)
            if Path(path).is_file():
                files = [path]
                fname = Path(fname).stem
            else:
                files = []
                for root, dirs, _files in os.walk(fpath):
                    for _file in _files:
                        root_file = os.path.join(root,_file)
                        if ftype in _file and all(s.strip() in root_file for s in fname.split("+")):
                            files.append(root_file)
        if not files:
            print("No file was selected")
        return files
