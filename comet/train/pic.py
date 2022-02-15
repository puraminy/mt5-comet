import curses as cur
import subprocess
import matplotlib.pyplot as plt
from curses import wrapper
import click
import numpy as np
from glob import glob
import six
import os
import seaborn as sns
from pathlib import Path
import pandas as pd
from comet.mycur.util import *
from mylogs import * 
import json
from comet.utils.myutils import *
import sys
from PIL import Image

def combine(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    return new_im

def get_files(dfpath, dfname):
    files = []
    if not dfname:
        fname = load_obj("dfname","","")
        if fname:
            dfname = [fname + ".png"]
    if not dfname:
        mlog.info("No file name provided")
    else:
        if len(dfname) > 1:
            files = list(dfname)
        else:
            dfname = dfname[0]
            path = os.path.join(dfpath, dfname)
            if Path(path).is_file():
                files = [path]
                dfname = Path(dfname).stem
            else:
                files = []
                for root, dirs, _files in os.walk(dfpath):
                    for _file in _files:
                        root_file = os.path.join(root, _file)
                        if all(s in root_file for s in dfname.split("|")):
                            files.append(root_file)
    if files:
        df = pd.DataFrame(columns={"name"})
        df["name"] = [Path(p).name for p in files]
        df["fname"] = files
        return df
    else:
        return None

def show_files(df):
    global dfname
    cmd = ""
    sel_row = 0
    ROWS, COLS = std.getmaxyx()
    ch = 1
    sel_row = 0
    left = 0
    max_row, max_col= text_win.getmaxyx()
    width = 15
    cond = ""
    sort = ""
    asc = False
    info_cols = load_obj("info_cols", "show_files", []) 
    sel_vals = []
    stats = []
    col_widths = load_obj("widths", "")
    def refresh():
        text_win.refresh(0, left, 0, 0, ROWS-1, COLS-2)
    def fill_text_win(rows):
        text_win.erase()
        for row in rows:
            mprint(row, text_win)
        refresh()
        get_key(std)


    #wwwwwwwwww
    main_df = df
    search_df = df
    consts = {}
    colors = ['blue','orange','green', 'red', 'purple', 'brown', 'pink','gray','olive','cyan']
    ax = None
    prev_cahr = ""
    sel_exp = ""
    sel_cols=["name"]
    sel_rows=[]
    in_search = False
    search = ""
    while ch != ord("q"):
        text_win.erase()
        left = min(left, max_col  - width)
        left = max(left, 0)
        sel_row = min(sel_row, len(df) - 1)
        sel_row = max(sel_row, 0)
        text = "{:<5}".format(sel_row)
        for i, sel_col in enumerate(sel_cols):
            if not sel_col in df:
                continue
            _w = col_widths[sel_col] if sel_col in col_widths else width
            head = textwrap.shorten(f"{i}) {sel_col}" , width=_w, placeholder=".")
            text += "{:<{x}}".format(head, x=_w) 
        mprint(text, text_win) 
        ii = 0
        top_margin = min(len(df), 5)
        #fffff
        infos = []
        sel_dict = {}
        for idx, row in df.iterrows():
           if ii < sel_row - top_margin:
               ii += 1
               continue
           text = "{:<5}".format(ii)
           for sel_col in sel_cols:
               if not sel_col in row:
                   continue
               content = str(row[sel_col])
               content = content.strip()
               _info = sel_col + ":" + content
               if sel_col in info_cols and ii == sel_row:
                    infos.append(_info)
               if ii == sel_row:
                   sel_dict[sel_col] = row[sel_col]
               _color = TEXT_COLOR
               _w = col_widths[sel_col] if sel_col in col_widths else width
               text += "{:<{x}}".format(content, x= _w)
           if ii in sel_rows:
               _color = HL_COLOR
           if ii == sel_row:
               _color = CUR_ITEM_COLOR

           mprint(text, text_win, color = _color) 
           ii += 1
           if ii > sel_row + ROWS - 4:
               break
        refresh()
        if in_search:
            consts["search"] = search
        for c in info_cols:
            if not c in df:
                continue
            if "score" in c:
                mean = df[c].mean()
                _info = f"Mean {c}:" + "{:.2f}".format(mean)
                infos.append(_info)
        infos.append("-------------------------")
        for key,val in consts.items():
            if type(val) == list:
                val = "-".join(val)
            infos.append("{:<5}:{}".format(key,val))
        infos.append("{:<5}:{}".format("len",str(len(df))))
        change_info(infos)

        prev_char = chr(ch)
        ch = get_key(std)
        char = chr(ch)
        vals = []
        get_cmd = False
        if ch == LEFT:
            in_search =False
            left -= width
        if ch == RIGHT:
            in_search =False
            left += width
        if ch == DOWN:
            in_search =False
            sel_row += 1
        elif ch == UP:
            in_search =False
            sel_row -= 1
        elif ch == cur.KEY_NPAGE:
            in_search =False
            sel_row += ROWS - 4
        elif ch == cur.KEY_PPAGE:
            in_search =False
            sel_row -= ROWS - 4
        elif is_enter(ch):
            in_search = False
        char = chr(ch)
        if in_search:
            if ch == cur.KEY_BACKSPACE or ch == 127:
                if len(search) > 1:
                    search = search[:-1]
                else:
                    search = ""
            else:
                search += char
            df = search_df[search_df["name"].str.contains(search, na=False)==True]
        else:
            search = ""
            search_df = df
            consts.pop("search", None)
            if char == "p":
                sel_pics = df.iloc[sel_rows]["fname"].tolist()
                images = [Image.open(x) for x in sel_pics]
                new_im = combine(images)
                pname = "/home/ahmad/heatmaps/out.png" 
                new_im.save(pname)
                if "ahmad" in home:
                    subprocess.run(["eog", pname])
            if char == " ":
                sel_rows.append(sel_row)
            if char == "D":
                df = df.iloc[sel_rows,:]
                sel_rows = []
            if char == "r":
                df = main_df
                sel_cols = ["name"]
                consts = {}
                info_cols = []
            if char == "/":
                in_search = True
                search_df = df
                consts["search"] = ""
                search = ""
            if char == ":":
                cmd = rowinput()
                if cmd == "q":
                    ch = ord("q")
                elif cmd:
                    df = get_files(dfpath, [cmd])

def biginput(prompt=":", default=""):
    rows, cols = std.getmaxyx()
    win = cur.newwin(12, cols - 10, 5, 5)
    _default = ""
    win.bkgd(' ', cur.color_pair(CUR_ITEM_COLOR))  # | cur.A_REVERSE)
    _comment, ret_ch = minput(win, 0, 0, "Enter text", 
            default=_default, mode =MULTI_LINE)
    if _comment == "<ESC>":
        _comment = ""
    return _comment

def rowinput(prompt=":", default=""):
    prompt = str(prompt)
    default = str(default)
    cmd, _ = minput(cmd_win, 0, 1, prompt, default=default, all_chars=True)
    if cmd == "<ESC>":
        cmd = ""
    return cmd


                
def change_info(infos):
    info_bar.erase()
    for msg in infos:
        mprint(str(msg), info_bar, color=HL_COLOR)
    rows,cols = std.getmaxyx()
    info_bar.refresh(0,0, rows -len(infos),0, rows-1, cols - 2)
si_hash = {}
def list_values(vals,si=0, sels=[]):
    tag_win = cur.newwin(15, 70, 3, 5)
    tag_win.bkgd(' ', cur.color_pair(TEXT_COLOR))  # | cur.A_REVERSE)
    tag_win.border()
    key = "_".join([str(x) for x in vals[:4]])
    if si == 0:
        if key in si_hash:
            si = si_hash[key]
    opts = {"items":{"sels":sels, "range":["Done!"] + vals}}
    is_cancled = True
    si,canceled, _ = open_submenu(tag_win, opts, "items", si, "Select a value", std)
    val = ""
    if not canceled and si > 0: 
        val = vals[si - 1]
        si_hash[key] = si
        is_cancled = False
    return is_cancled, val


text_win = None
info_bar = None
cmd_win = None
main_win = None
text_width = 60
std = None
dfname = ""
dfpath = ""
base_dir = os.path.join(resPath, "sel")
def start(stdscr):
    global info_bar, text_win, cmd_win, std, main_win, colors, dfname
    stdscr.refresh()
    std = stdscr
    now = datetime.datetime.now()
    rows, cols = std.getmaxyx()
    height = rows - 1
    width = cols
    # mouse = cur.mousemask(cur.ALL_MOUSE_EVENTS)
    text_win = cur.newpad(rows * 50, cols * 20)
    text_win.bkgd(' ', cur.color_pair(TEXT_COLOR)) # | cur.A_REVERSE)
    cmd_win = cur.newwin(1, cols, rows - 1, 0)
    info_bar = cur.newpad(rows*10, cols*20)
    info_bar.bkgd(' ', cur.color_pair(HL_COLOR)) # | cur.A_REVERSE)

    cur.start_color()
    cur.curs_set(0)
    # std.keypad(1)
    cur.use_default_colors()

    colors = [str(y) for y in range(-1, cur.COLORS)]
    if cur.COLORS > 100:
        colors = [str(y) for y in range(-1, 100)] + [str(y) for y in range(107, cur.COLORS)]


    theme = {'preset': 'default', "sep1": "colors", 'text-color': '247', 'back-color': '233', 'item-color': '71','cur-item-color': '251', 'sel-item-color': '33', 'title-color': '28', "sep2": "reading mode",           "dim-color": '241', 'bright-color':"251", "highlight-color": '236', "hl-text-color": "250", "inverse-highlight": "True", "bold-highlight": "True", "bold-text": "False", "input-color":"234", "sep5": "Feedback Colors"}

    reset_colors(theme)
    df = get_files(dfpath, dfname)
    if df is not None:
        show_files(df)
    else:
        mlog.info("No tsv or json file was found")

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
    default="parent",
    type=str,
    help=""
)
def main(fname, path, fid):
    global dfname,dfpath,file_id
    file_id = fid
    if fname != "last":
        dfname = fname 
        dfpath = path
    set_app("show_files")
    wrapper(start)

if __name__ == "__main__":
    main()
