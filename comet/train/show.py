import curses as cur
from curses import wrapper
import click
import numpy as np
import os
from pathlib import Path
import pandas as pd
from nodcast.util.util import *
from mylogs import * 
import json
from comet.utils.myutils import *

def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    sd = superitems(data)
    fname = Path(path).stem
    if fname == "results":
        main_df = pd.DataFrame(sd, columns=["exp","model","lang", "method","wrap","frozen","epochs","stype", "date", "dir", "score"])
    else:
        main_df = pd.DataFrame(sd, columns=["tid","exp","model","lang", "method","wrap","frozen","epochs","date", "field", "text"])

    out = f"{fname}.tsv"
    df = main_df.pivot(index=list(main_df.columns[~main_df.columns.isin(['field', 'text'])]), columns='field').reset_index()

    #df.columns = list(map("".join, df.columns))
    df.columns = [('_'.join(str(s).strip() for s in col if s)).replace("text_","") for col in df.columns]
    df.to_csv(os.path.join(resPath, out), sep="\t", index = False)
    return df

def show_df(df):
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
    info_cols = load_obj("info_cols", dfname, []) 
    sel_vals = []
    stats = []
    col_widths = load_obj("widths", "")
    def refresh():
        text_win.refresh(0, left, 1, 1, ROWS-1, COLS-2)
    def fill_text_win(rows):
        text_win.erase()
        for row in rows:
            mprint(row, text_win)
        refresh()
        get_key(std)

    if not col_widths:
        col_widths = {"qid":5, "model":30, "pred_text1":30, "epochs":30, "date":30, "rouge_score":7, "bert_score":7, "input_text":50}

    store_back = False
    df['id']=df.index
    df = df.reset_index(drop=True)
    df['input_text'] = df['input_text'].str.replace('##','')
    df['input_text'] = df['input_text'].str.strip()
    main_df = df
    edit_col = ""
    count_col = ""
    consts = {}
    save_obj(dfname, "dfname", "")
    sel_cols = load_obj("sel_cols", dfname)
    if not sel_cols:
        sel_cols = list(df.columns)
    for col in df.columns:
        if "score" in col:
            df[col] = pd.to_numeric(df[col])
    fav_path = os.path.join(base_dir, dfname + "_fav.tsv")
    if Path(fav_path).exists():
        fav_df = pd.read_table(fav_path)
    else:
        fav_df = pd.DataFrame(columns = df.columns)
    sel_path = os.path.join(base_dir, dfname + "_sel.tsv")
    if Path(sel_path).exists():
        sel_df = pd.read_table(sel_path)
    else:
        sel_df = pd.DataFrame(columns = df.columns)
    back = {"df":df, "sel_cols":sel_cols, "info_cols":info_cols, "sel_row":0}
    filter_df = main_df
    #wwwwwwwwww
    prev_cahr = ""
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
               if "score" in sel_col:
                   content = "{:.2f}".format(float(content))
               _info = sel_col + ":" + content
               if sel_col in info_cols and ii == sel_row:
                    infos.append(_info)
               if ii == sel_row:
                   sel_dict[sel_col] = row[sel_col]
               _color = TEXT_COLOR
               _w = col_widths[sel_col] if sel_col in col_widths else width
               text += "{:<{x}}".format(content, x= _w)

           if ii == sel_row:
                _color = CUR_ITEM_COLOR
           mprint(text, text_win, color = _color) 
           ii += 1
           if ii > sel_row + ROWS - 4:
               break
        refresh()
        for c in info_cols:
            if not c in df:
                continue
            if "score" in c:
                mean = df[c].mean()
                _info = f"Mean {c}:" + "{:.2f}".format(mean)
                infos.append(_info)
        infos.append("-------------------------")
        consts["len"] = str(len(df))
        for key,val in consts.items():
            if type(val) == list:
                val = "-".join(val)
            infos.append("{:<5}:{}".format(key,val))
        change_info(infos)

        if store_back:
            back = {"df":df, "sel_cols":sel_cols, "info_cols":info_cols, "sel_row":sel_row}
        prev_char = chr(ch)
        ch = get_key(std)
        store_back = False
        char = chr(ch)
        vals = []
        get_cmd = False
        if ch == LEFT:
            left -= width
        if ch == RIGHT:
            left += width
        if ch == DOWN:
            sel_row += 1
        elif ch == UP:
            sel_row -= 1
        elif ch == cur.KEY_NPAGE:
            sel_row += ROWS - 4
        elif ch == cur.KEY_PPAGE:
            sel_row -= ROWS - 4
        elif char in ["l","L"]:
            cmd, _ = minput(cmd_win, 0, 1, "File Name=", default=dfname, all_chars=True)
            if cmd != "<ESC>":
                dfname = cmd
                path = os.path.join(base_dir, dfname + ".tsv")
                if not Path(path) or char == "L":
                    path = os.path.join(base_dir, dfname + ".json")
                    df = load_results(path)
                    sel_cols = list(df.columns) 
                else:
                    df = pd.read_table(path)
                    sel_cols = load_obj("sel_cols", dfname, list(df.columns)) 
                save_obj(dfname, "dfname", "")
                info_cols = load_obj("info_cols", dfname, []) 
                main_df = df
        elif char in list("0123456789"):
            canceled, col, val = list_df_values(df, get_val=False)
            if not canceled:
                sel_cols = order(sel_cols, [col],int(char))
        elif char in ["e","E"]:
            if not edit_col or char == "E":
                canceled, col, val = list_df_values(df, get_val=False)
                if not canceled:
                    edit_col = col
                    consts["edit col"] = edit_col
                    refresh()
            if edit_col:
                new_val = rowinput()
                if new_val:
                    df.at[sel_row, edit_col] = new_val
                    char = "SS"
        elif char in ["a", "A"]:
            canceled, col, val = list_df_values(df, get_val=False)
            if not canceled:
                if not col in sel_cols:
                    sel_cols.insert(0, col)
                    save_obj(sel_cols, "sel_cols", dfname)
        elif char in ["i", "I"]:
            canceled, col, val = list_df_values(df, get_val=False)
            if not canceled:
                if not col in info_cols: info_cols.append(col)
                save_obj(info_cols, "info_cols", dfname)
                if char == "I":
                    sel_cols.remove(col)
        elif char == "x":
            sel_df = sel_df.append(df.iloc[sel_row])
            mbeep()
            sel_df.to_csv(sel_path, sep="\t", index=False)
        elif char == "X":
            back["df"] = df
            back["sel_cols"] = sel_cols
            back["info_cols"] = info_cols
            back["sel_row"] = sel_row
            df = sel_df
        elif char == "z":
            fav_df = fav_df.append(df.iloc[sel_row])
            mbeep()
            fav_df.to_csv(fav_path, sep="\t", index=False)
        elif char == "Z":
            back["df"] = df
            back["sel_cols"] = sel_cols
            back["info_cols"] = info_cols
            back["sel_row"] = sel_row
            df = fav_df
        elif char == "j":
            canceled, col = list_values(info_cols)
            if not canceled:
                pos = rowinput("pos:","")
                if pos:
                    info_cols.remove(col)
                    if int(pos) > 0:
                        info_cols.insert(int(pos), col)
                    else:
                        sel_cols.insert(0, col)
                    save_obj(info_cols, "info_cols", dfname)
                    save_obj(sel_cols, "sel_cols", dfname)
        elif char == "s":
            canceled, col, val = list_df_values(df, get_val=False)
            if not canceled:
                if col == sort:
                    asc = not asc
                sort = col
                df = df.sort_values(by=sort, ascending=asc)
                if not prev_char in ["g","G"]:
                    sel_cols = order(sel_cols, [sort])
        elif char in ["c","C"]: 
            counts = {}
            for col in df:
               counts[col] = df[col].nunique()
            df = pd.DataFrame(data=[counts], columns = df.columns)
        elif char == "U":
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        elif char == "u": 
            if not count_col:
                canceled, col, _ = list_df_values(df, get_val=False)
                if not canceled:
                    count_col = col
                    consts["count col"] = col
            if count_col:
                df = df[col].value_counts(ascending=False).reset_index()
                sel_cols = list(df.columns)
                col_widths["index"]=50
                info_cols = []
        elif char == "T":
            df = df.drop_duplicates(['prefix'])
        elif char == "g": 
            score_col = "rouge_score"
            group_col = "pred_text1"
            df = df.sort_values(score_col, ascending=False).\
                 drop_duplicates(['date','prefix','input_text']).\
                    rename(columns={group_col:'top_target'}).\
                      merge(df.groupby(['prefix','input_text'],as_index=False)[group_col].agg('<br />'.join))
            if not group_col in info_cols: info_cols.append(group_col)
        elif char == "G":
            canceled, col, _ = list_df_values(df, get_val=False)
            if not canceled:
               g_cols = [col, "method", "model", "rouge_score","bert_score"]
               sel_cols = df[g_cols]
               df = df.groupby(col).agg({"rouge_score":"mean","bert_score":"mean",
                                         "method":"first","model":"first"})
               df = df.reset_index()
               sel_cols = order(sel_cols, g_cols)
        elif char == "D":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                del main_df[col]
                char = "SS"
                if col in df:
                    del df[col]

        elif char in ["d"]:
            canceled, col, val = list_df_values(main_df)
            if not canceled:
                main_df = main_df.drop(main_df[main_df[col] == val].index)
                char = "SS"
                info_cols = []
                if col in df:
                    df = df.drop(df[df[col] == val].index)
        elif ch == cur.KEY_DC:
            col = sel_cols[0]
            val = sel_dict[col]
            cmd = rowinput("Are you sure you want to delete {} == {} ".format(col,val))
            if cmd == "y":
                main_df = main_df.drop(main_df[main_df[col] == val].index)
                char = "SS"
                info_cols = []
                if col in df:
                    df = df.drop(df[df[col] == val].index)
        elif char in ["m","M"]:
            cond = ""
            canceled = False
            sels = []
            info_cols = []
            sel_cols = []
            while not canceled:
                canceled, col, val = list_df_values(main_df, col="model", get_val=True,sels=sels)
                cond += f"| (df['{col}'] == '{val}') "
                info_cols.append("input_text_"+val)
                info_cols.append("prefix_"+val)
                sel_cols.append("pred_text1_"+val)
                sels.append(val)
            cond = cond.strip("|")
            df = main_df[eval(cond)]
            if df.duplicated(['qid','model']).any():
                show_err("There is duplicated rows for qid and model")
                char = "r"
            else:
                df = df.set_index(['qid','model'])[['pred_text1', 'input_text','prefix']].unstack()
                df.columns = list(map("_".join, df.columns))
                for s in sel_cols:
                    col_widths[s] = 35
                store_back = True
        elif char in ["f","F"]:
            canceled, col, val = list_df_values(main_df)
            if not canceled:
                df = filter_df[filter_df[col] == val]
                df = df.reset_index()
                if not "filter" in consts:
                    consts["filter"] = []
                consts["filter"].append(" {} == {}".format(col,val))
                if char == "F":
                    sel_cols = order(sel_cols, [col])
                sel_row = 0
                filter_df = df
        elif is_enter(ch):
            col = sel_cols[0]
            val = sel_dict[col]
            if not "filter" in consts:
                consts["filter"] = []
            consts["filter"].append("{} == {}".format(col,val))
            df = filter_df[filter_df[col] == val]
            df = df.reset_index()
            if char == "F":
                sel_cols = order(sel_cols, [col])
            sel_row = 0
            filter_df = df
        elif char == "w":
            canceled, col, val = list_df_values(df, get_val=False)
            if not canceled:
                cmd, _ = minput(cmd_win, 0, 1, ":width=", all_chars=True)
                if cmd.isnumeric():
                    col_widths[col] = int(cmd)
                    save_obj(col_widths, "widths", "")
        elif char == "/":
            search = rowinput("/")
            mask = np.column_stack([df[col].str.contains(search, na=False) for col in df])
            sel_row = df.loc[mask.any(axis=1)].index[0]
        elif char == ":":
            cmd = rowinput()
            if cmd.isnumeric():
                sel_row = int(cmd)
            elif cmd == "q":
                ch = ord("q")
        elif char == "q":
            cmd = rowinput("Are you sure you want to exit? (y/n)")
            if cmd == "y":
                ch = ord("q")
            else:
                ch = 0

        elif not char in ["q", "S","r"]:
            pass
            #mbeep()
        if char == "S":
            cmd, _ = minput(cmd_win, 0, 1, "File Name=", default=dfname, all_chars=True)
            if cmd != "<ESC>":
                dfname = cmd
                char = "SS"
        if char == "SS":
                main_df.to_csv(os.path.join(base_dir, dfname+".tsv"), sep="\t", index=False)
                save_obj(dfname, "dfname", dfname)
        if char == "r":
            df = main_df
            sel_cols = list(df.columns)
            save_obj(sel_cols,"sel_cols",dfname)
            info_cols = []
        if char == "b" and back:
            df = back["df"] 
            consts["filter"] = []
            sel_cols = back["sel_cols"] 
            info_cols = back["info_cols"]
            sel_row = back["sel_row"]

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
    cmd, _ = minput(cmd_win, 0, 1, prompt, default=default, all_chars=True)
    if cmd == "<ESC>":
        cmd = ""
    return cmd

def order(sel_cols, cols, pos=0):
    z = [item for item in sel_cols if item not in cols] 
    z[pos:pos] = cols
    save_obj(z, "sel_cols",dfname)
    return z


                
def change_info(infos):
    info_bar.erase()
    for msg in infos:
        mprint(str(msg), info_bar, color=HL_COLOR)
    rows,cols = std.getmaxyx()
    info_bar.refresh(0,0, rows -len(infos),0, rows-1, cols)
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

def list_df_values(df, col ="", get_val=True,si=0,vi=0, sels=[]):
    is_cancled = False
    if not col:
        cols = list(df.columns)
        is_cancled, col = list_values(cols,si, sels)
    val = ""
    if col and get_val and not is_cancled:
        vals = list(df[col].unique())
        is_cancled, val = list_values(vals,vi, sels)
    return is_cancled, col, val 


text_win = None
info_bar = None
cmd_win = None
main_win = None
text_width = 60
std = None
dfname = ""
base_dir = resPath
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
    info_bar = cur.newpad(20, cols -2)
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
    if not dfname:
        fname = load_obj("dfname","","")
        dfname = fname + ".tsv"
    if not dfname:
        mlog.info("No file name provided")
    else:
        path = os.path.join(base_dir, dfname)
        dfname = Path(path).stem
        if not Path(path).exists():
            mlog.info("File %s doesn't exists!", path)
        if path.endswith(".tsv"):
            df = pd.read_table(path)
            show_df(df)
        elif path.endswith(".json"):
            df = load_results(path)
            show_df(df)
        else:
            mlog.info("No tsv or json file was found")

@click.command()
@click.argument("fname", type=str)
def main(fname):
    global dfname
    if fname != "last":
        dfname = fname
    set_app("showdf")
    wrapper(start)

if __name__ == "__main__":
    main()
