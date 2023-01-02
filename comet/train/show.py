import curses as cur
import subprocess
from functools import reduce
import matplotlib.pyplot as plt
from curses import wrapper
import click
import numpy as np
from glob import glob
import six
import debugpy
import os, shutil
import re
import seaborn as sns
from pathlib import Path
import pandas as pd
from comet.mycur.util import *
from mylogs import * 
import json
from comet.utils.myutils import *
file_id = "name"
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def combine_x(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    return new_im

def combine_y(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = max(widths)
    max_height = sum(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    y_offset = 0
    for im in images:
      new_im.paste(im, (0, y_offset))
      y_offset += im.size[1]

    return new_im

def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    sd = superitems(data)
    fname = Path(path).stem
    if fname == "results":
        main_df = pd.DataFrame(sd, columns=["exp","model","lang", "template","wrap","frozen","epochs","stype", "date", "dir", "score"])
    else:
        main_df = pd.DataFrame(sd, columns=["tid","exp","model","lang", "template","wrap","frozen","epochs","date", "field", "text"])

    out = f"{fname}.tsv"
    df = main_df.pivot(index=list(main_df.columns[~main_df.columns.isin(['field', 'text'])]), columns='field').reset_index()

    #df.columns = list(map("".join, df.columns))
    df.columns = [('_'.join(str(s).strip() for s in col if s)).replace("text_","") for col in df.columns]
    df.to_csv(path.replace("json", "tsv"), sep="\t", index = False)
    return df

def find_common(df, main_df, on_col_list, s_rows, FID, char):
    dfs_items = [] 
    dfs = []
    ii = 0
    dfs_val = {}
    for s_row in s_rows:
        exp=df.iloc[s_row]["exp_id"]
        prefix=df.iloc[s_row]["prefix"]
        dfs_val["exp" + str(ii)] = exp
        mlog.info("%s == %s", FID, exp)
        cond = f"(main_df['{FID}'] == '{exp}') & (main_df['prefix'] == '{prefix}')"
        tdf = main_df[(main_df[FID] == exp) & (main_df['prefix'] == prefix)]
        tdf = tdf[["pred_text1", "exp_name", "id","hscore", "bert_score","query", "resp", "template", "rouge_score", "fid","prefix", "input_text","target_text", "sel"]]
        tdf = tdf.sort_values(by="rouge_score", ascending=False)
        if len(tdf) > 1:
            tdf = tdf.groupby(on_col_list).agg({"exp_name":"first","query":"first", "resp":"first","input_text":"first","target_text":"first", "hscore":"first", "template":"first", "rouge_score":"first","prefix":"first","pred_text1":"first", "fid":"first", "id":"count","bert_score":"first", "sel":"first"}).reset_index(drop=True)
            for on_col in on_col_list:
                tdf[on_col] = tdf[on_col].astype(str).str.strip()
            #tdf = tdf.set_index(on_col_list)
        dfs.append(tdf) #.copy())
        ii += 1
    if ii > 1:
        intersect = reduce(lambda  left,right: pd.merge(left,right,on=on_col_list,
                                    how='inner'), dfs)
        if char == "n":
            union = reduce(lambda  left,right: pd.merge(left,right,on=on_col_list,
                                    how='outer'), dfs)
            dfs_val["union"] = str(len(union))
            dfs_val["int"] = str(len(intersect))
            dfs_items.append(dfs_val)
            df = pd.DataFrame(dfs_items)
        else:
            df = intersect
    else:
       df = tdf
       df["sum_fid"] = df["id"].sum()
    return df, exp

def show_df(df):
    global dfname
    cmd = ""
    sel_row = 0
    cur_col = 0
    ROWS, COLS = std.getmaxyx()
    ch = 1
    sel_row = 0
    left = 0
    max_row, max_col= text_win.getmaxyx()
    width = 15
    top = 10
    height = 10
    cond = ""
    sort = ""
    asc = False
    info_cols = load_obj("info_cols", dfname, []) 
    info_cols_back = []
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

    def save_df(df): 
        return
        s_rows = range(len(df))
        show_msg("Saving ...")
        for s_row in s_rows:
            exp=df.iloc[s_row]["exp_id"]
            tdf = main_df[main_df["fid"] == exp]
            spath = tdf.iloc[0]["path"]
            tdf.to_csv(spath, sep="\t", index=False)


    if not col_widths:
        col_widths = {"query":50, "model":30, "pred_text1":30, "epochs":30, "date":30, "rouge_score":7, "bert_score":7, "input_text":50}

    df['id']=df.index
    df = df.reset_index(drop=True)
    if not "tag" in df:
        df["tag"] = np.NaN 

    if not "hscore" in df:
        df["hscore"] = np.NaN 

    if not "pid" in df:
        df["pid"] = 0
    if not "l1_decoder" in df:
        df["l1_decoder"] ="" 
        df["l1_encoder"] ="" 
        df["cossim_decoder"] ="" 
        df["cossim_encoder"] ="" 

    if not "query" in df:
        df["query"] = df["input_text"]
    if not "learning_rate" in df:
        df["learning_rate"] = 1

    if not "prefixed" in df:
        df["prefixed"] = False

    if not "sel" in df:
       df["sel"] = False

    if not "bert_score" in df:
       df["bert_score"] = 0

    if "exp_id" in df:
        df = df.rename(columns={"exp_id":"expid"})

    if "input_text" in df:
        df['input_text'] = df['input_text'].str.replace('##','')
        df['input_text'] = df['input_text'].str.split('>>').str[0]
        df['input_text'] = df['input_text'].str.strip()
    main_df = df
    edit_col = ""
    count_col = ""
    extra = {"filter":[], "inp":""}
    save_obj(dfname, "dfname", "")
    sel_cols = list(df.columns)
    for col in df.columns:
        if col.endswith("score"):
            df[col] = pd.to_numeric(df[col])
    fav_path = os.path.join(base_dir, dfname + "_fav.tsv")
    if Path(fav_path).exists():
        fav_df = pd.read_table(fav_path)
    else:
        fav_df = pd.DataFrame(columns = df.columns)
    sel_path = os.path.join(base_dir, "test.tsv")
    if Path(sel_path).exists():
        sel_df = pd.read_table(sel_path)
    else:
        sel_df = pd.DataFrame(columns = ["prefix","input_text","target_text"])

    back = []
    sels = []
    filter_df = main_df
    if "src_path" in df:
        sel_path = df.loc[0, "src_path"]
        if not sel_path.startswith("/"):
            sel_path = os.path.join(home, sel_path)
    if "pred_text1" in df:
        br_col = df.loc[: , "bert_score":"rouge_score"]
        df['nr_score'] = df['rouge_score']
        df['nr_score'] = np.where((df['bert_score'] > 0.3) & (df['nr_score'] < 0.1), df['bert_score'], df['rouge_score'])

    #wwwwwwwwww
    colors = ['blue','orange','green', 'red', 'purple', 'brown', 'pink','gray','olive','cyan']
    contexts = {"g":"grouped", "X":"view", "r":"main"}
    ax = None
    context = dfname
    font = ImageFont.truetype("/usr/share/vlc/skins2/fonts/FreeSans.ttf", 68)
    seq = ""
    search = ""
    on_col_list = []
    sel_fid = "" 
    open_dfnames = [dfname]
    #if not "learning_rate" in df:
    #    df[['fid_no_lr', 'learning_rate']] = df['fid'].str.split('_lr_', 1, expand=True)
    if not "plen" in df:
        df["plen"] = 8
    if not "blank" in df:
        df["blank"] = "blank"
    if not "opt_type" in df:
        df["opt_type"] = "na"
    if not "rouge_score" in df:
        df["rouge_score"] = 0
    if not "bert_score" in df:
        df["bert_score"] = 0
    prev_cahr = ""
    FID = "fid"
    hotkey = "gG"
    sel_exp = ""
    infos = []
    back_rows = []
    sel_rows = []
    cmd = ""
    prev_cmd = ""

    def row_print(df, sel_row, col_widths ={}, _print=False):
        ii = 0
        infos = []
        top_margin = min(len(df), 5)
        sel_dict = {}
        for idx, row in df.iterrows():
           if ii < sel_row - top_margin:
               ii += 1
               continue
           text = "{:<5}".format(ii)
           _sels = []
           _infs = []
           _color = TEXT_COLOR
           if ii in sel_rows:
               _color = HL_COLOR
           if ii == sel_row:
                _color = CUR_ITEM_COLOR
           if _print:
               mprint(text, text_win, color = _color, end="") 
           if _print:
               _cols = sel_cols + info_cols
           else:
               _cols = sel_cols
           for sel_col in _cols: 
               if  sel_col in _sels:
                   continue
               if not sel_col in row: 
                   if sel_col in sel_cols:
                       sel_cols.remove(sel_col)
                   continue
               content = str(row[sel_col])
               content = content.strip()
               if "score" in sel_col:
                   try:
                       content = "{:.2f}".format(float(content))
                   except:
                       pass
               _info = sel_col + ":" + content
               if sel_col in info_cols:
                   if ii == sel_row and not sel_col in _infs:
                      infos.append(_info)
                      _infs.append(sel_col)
               if ii == sel_row:
                   sel_dict[sel_col] = row[sel_col]
               if not sel_col in col_widths:
                   col_widths[sel_col] = len(content) + 4
               if len(content) > col_widths[sel_col]:
                   col_widths[sel_col] = len(content) + 4
               col_widths[sel_col] = min(col_widths[sel_col],40)
               _w = col_widths[sel_col] 
               if sel_col in sel_cols:
                   if cur_col < len(sel_cols) and sel_col == sel_cols[cur_col]:
                       if ii == sel_row:
                          cell_color = HL_COLOR
                       else:
                           cell_color = TITLE_COLOR
                   else:
                       cell_color = _color
                   text = textwrap.shorten(text, width=36, placeholder="...")
                   text = "{:<{x}}".format(content, x= _w)
                   if _print:
                       mprint(text, text_win, color = cell_color, end="") 
                   _sels.append(sel_col)

           _end = "\n"
           if _print:
               mprint("", text_win, color = _color, end="\n") 
           ii += 1
           if ii > sel_row + ROWS - 4 - len(infos):
               break
        return infos, col_widths

    def backit(df, sel_cols):
        back.append(df)
        sels.append(sel_cols.copy())
        back_rows.append(sel_row)
    for _col in ["input_text","pred_text1","target_text"]:
        if _col in df:
            df[_col] = df[_col].astype(str)

    map_cols = {
            "epochs_num":"epn",
            "exp_trial":"exp",
            "template":"tn",
            }
    adjust = True
    show_consts = True
    show_extra = False
    consts = {}
    extra = {"filter":[]}
    while ch != ord("q"):
        text_win.clear()
        left = min(left, max_col  - width)
        left = max(left, 0)
        top = min(top, max_row  - height)
        top = max(top, 0)
        sel_row = min(sel_row, len(df) - 1)
        sel_row = max(sel_row, 0)
        cur_col = min(cur_col, len(sel_cols) - 1)
        cur_col = max(cur_col, 0)
        if not hotkey:
            if adjust:
                _, col_widths = row_print(df, sel_row, col_widths={})
            text = "{:<5}".format(sel_row)
            for i, sel_col in enumerate(sel_cols):
               if not sel_col in df:
                   sel_cols.remove(sel_col)
                   continue
               head = sel_col if not sel_col in map_cols else map_cols[sel_col] 
               #head = textwrap.shorten(f"{i} {head}" , width=15, placeholder=".")
               if not sel_col in col_widths and not adjust:
                    _, col_widths = row_print(df, sel_row, col_widths={})
                    adjust = True
               if sel_col in col_widths and len(head) + 5 > col_widths[sel_col]:
                   col_widths[sel_col] = len(head) + 5
               if sel_col in col_widths:
                   _w = col_widths[sel_col] 
               text += "{:<{x}}".format(head, x=_w) 
            mprint(text, text_win) 
            #fffff
            infos,_ = row_print(df, sel_row, col_widths, True)
            refresh()
        for c in info_cols:
            if not c in df:
                continue
            if "score" in c:
                mean = df[c].mean()
                _info = f"Mean {c}:" + "{:.2f}".format(mean)
                infos.append(_info)
        infos.append("-------------------------")
        if show_consts:
            consts["len"] = str(len(df))
            for key,val in consts.items():
                if type(val) == list:
                    val = "-".join(val)
                infos.append("{:<5}:{}".format(key,val))
        if show_extra:
            show_extra = False
            for key,val in extra.items():
                if type(val) == list:
                    val = "-".join(val)
                infos.append("{:<5}:{}".format(key,val))
        change_info(infos)

        prev_char = chr(ch)
        prev_cmd = cmd
        cmd = ""
        if hotkey == "":
            ch = std.getch()
        else:
            ch, hotkey = ord(hotkey[0]), hotkey[1:]
        char = chr(ch)
        extra["inp"] = char

        seq += char
        vals = []
        get_cmd = False
        adjust = True
        context = contexts[char] if char in contexts else ""
        if ch == SLEFT:
            left -= 10
            adjust = False
        if ch == SRIGHT:
            left += 10
            adjust = False
        if ch == SDOWN:
            info_cols_back = info_cols.copy()
            info_cols = []
        if ch == SUP:
            info_cols = info_cols_back.copy()
        if ch == LEFT:
            cur_col -= 1
            cur_col = max(0, cur_col)
            width = col_widths[sel_cols[cur_col]]
            _sw = sum([col_widths[x] for x in sel_cols[:cur_col]])
            if _sw < left:
                left = _sw - width - 10 
            adjust = False
        if ch == RIGHT:
            cur_col += 1
            cur_col = min(len(sel_cols)-1, cur_col)
            width = col_widths[sel_cols[cur_col]]
            _sw = sum([col_widths[x] for x in sel_cols[:cur_col]])
            if _sw >= left + COLS - 10:
                left = _sw - 10 
            adjust = False
        if char in ["+","-","*","/"]:
            _inp=df.iloc[sel_row]["input_text"]
            _prefix=df.iloc[sel_row]["prefix"]
            _pred_text=df.iloc[sel_row]["pred_text1"]
            _fid=df.iloc[sel_row]["fid"]
            cond = ((main_df["fid"] == _fid) & (main_df["input_text"] == _inp) &
                    (main_df["prefix"] == _prefix) & (main_df["pred_text1"] == _pred_text))
            if char == "+": _score = 1.
            if char == "-": _score = 0.
            if char == "/": _score = 0.4
            if char == "*": _score = 0.7

            main_df.loc[cond, "hscore"] = _score 
            sel_exp = _fid
            sel_row += 1
            adjust = False
        if ch == DOWN:
            if context == "inp":
                back_rows[-1] += 1
                hotkey = "bp"
            else:
                sel_row += 1
                adjust = False
        elif ch == UP: 
            if context == "inp":
                back_rows[-1] -= 1
                hotkey = "bp"
            else:
                sel_row -= 1
                adjust = False
        elif ch == cur.KEY_NPAGE:
            sel_row += ROWS - 4
        elif ch == cur.KEY_HOME:
            sel_row = 0 
        elif ch == cur.KEY_SHOME:
            left = 0 
        elif ch == cur.KEY_END:
            sel_row = len(df) -1
        elif ch == cur.KEY_PPAGE:
            sel_row -= ROWS - 4
        elif char == "l" and prev_char == "l":
            seq = ""
        elif char == "=" and prev_char == "x":
            col = info_cols[-1]
            sel_cols.insert(cur_col, col)
        elif char == ">":
            col = info_cols.pop()
            sel_cols.insert(cur_col, col)
        elif char in "01234" and prev_char == "#":
            canceled, col, val = list_df_values(df, get_val=False)
            if not canceled:
                sel_cols = order(sel_cols, [col],int(char))
        elif char in ["e","E"]:
            if not edit_col or char == "E":
                canceled, col, val = list_df_values(df, get_val=False)
                if not canceled:
                    edit_col = col
                    extra["edit col"] = edit_col
                    refresh()
            if edit_col:
                new_val = rowinput()
                if new_val:
                    df.at[sel_row, edit_col] = new_val
                    char = "SS"
        elif char in ["%"]:
            canceled, col, val = list_df_values(df, get_val=False)
            if not canceled:
                if not col in sel_cols:
                    sel_cols.insert(0, col)
                    save_obj(sel_cols, "sel_cols", context)
        elif char in ["W"] and prev_char == "x":
            save_df(df)
        elif char in ["B", "N"]:
            s_rows = sel_rows
            from comet.train.eval import do_score
            if not s_rows:
                s_rows = [sel_row]
            if char == "N":
                s_rows = range(len(df))
            for s_row in s_rows:
                exp=df.iloc[s_row]["exp_id"]
                _score=df.iloc[s_row]["bert_score"]
                #if _score > 0:
                #    continue
                cond = f"(main_df['{FID}'] == '{exp}')"
                tdf = main_df[main_df[FID] == exp]
                #df = tdf[["pred_text1", "id", "bert_score","query", "template", "rouge_score", "fid","prefix", "input_text","target_text"]]
                spath = tdf.iloc[0]["path"]
                spath = str(Path(spath).parent)
                tdf = do_score(tdf, "rouge-bert", spath, reval=True) 
                tdf = tdf.reset_index()
                #main_df.loc[eval(cond), "bert_score"] = tdf["bert_score"]
            df = main_df
            hotkey = "gG"
        elif char == "l":
            exp=df.iloc[sel_row]["expid"]
            exp = str(exp)
            logs = glob(str(exp) + "*.log")
            if logs:
                log = logs[0]
                with open(log,"r") as f:
                    infos = f.readlines()
                ii = 0
                inf = infos[ii:ii+30]
                change_info(inf)
                cc = std.getch()
                while chr(cc) != "b":
                    if cc == DOWN:
                        ii += 1
                    if cc == UP:
                        ii -= 1
                    if cc == cur.KEY_NPAGE:
                        ii += 10
                    if cc == cur.KEY_PPAGE:
                        ii -= 10
                    if cc == cur.KEY_HOME:
                        ii = 0
                    if cc == cur.KEY_END:
                        ii = len(infos) - 20 
                    ii = max(ii, 0)
                    ii = min(ii, len(infos)-10)
                    inf = infos[ii:ii+30]
                    change_info(inf)
                    cc = std.getch()

        elif char == "<":
            col = sel_cols[cur_col]
            sel_cols.remove(col)
            info_cols.append(col)
            save_obj(sel_cols, "sel_cols", context)
            save_obj(info_cols, "info_cols", context)
        elif char == "X" and not prev_char == "x":
            backit(df,sel_cols)
            exp=df.iloc[sel_row]["exp_id"]
            cond = f"(main_df['{FID}'] == '{exp}')"
            df = main_df[main_df[FID] == exp]
            sel_cols=["fid","input_text","target_text","pred_text1", "hscore","bert_score","prefix"]
            df = df[sel_cols]
            df = df.sort_values(by="input_text", ascending=False)
        elif char in ["I"] or ch == cur.KEY_IC:
            if char == "I":
                canceled, col, val = list_df_values(df, get_val=False)
            else:
                canceled, col, val = list_df_values(main_df, get_val=False)
            if not canceled:
                if not col in sel_cols: 
                    sel_cols.insert(cur_col, col)
                else:
                    sel_cols.remove(col)
                    sel_cols.insert(cur_col, col)
                save_obj(sel_cols, "sel_cols", context)
                if col in info_cols:
                    info_cols.remove(col)
                    save_obj(info_cols, "info_cols", context)
        elif char in ["o","O"]:
            inp = df.loc[sel_row,["prefix", "input_text"]]
            df.loc[(df.prefix == inp.prefix) & 
                    (df.input_text == inp.input_text), 
                    ["sel"]] = False
            sel_df.loc[(sel_df.prefix == inp.prefix) & 
                    (sel_df.input_text == inp.input_text), 
                    ["sel"]] = False
            df = df.sort_values(by="sel", ascending=False)
            consts["sel_path"] = sel_path + "|"+ str(len(sel_df)) + "|" + str(sel_df["input_text"].nunique())
            mbeep()
            df = df.sort_values(by="sel", ascending=False).reset_index(drop=True)
            row = df.loc[(df.prefix == inp.prefix) & 
                    (df.input_text == inp.input_text),:]
            sel_row = row.index[0]
            sel_df = sel_df.sort_values(by=["prefix","input_text","target_text"]).drop_duplicates()
            sel_df.to_csv(sel_path, sep="\t", index=False)
        elif char in ["w","W"]:
            inp = df.loc[sel_row,["prefix", "input_text"]]
            df.loc[(df.prefix == inp.prefix) & 
                    (df.input_text == inp.input_text),["sel"]] = True
            _rows = main_df.loc[(main_df.prefix == inp.prefix) & 
                    (main_df.input_text == inp.input_text), 
                    ["prefix", "input_text", "target_text"]].drop_duplicates()
            sel_df = sel_df.append(_rows)
            df = df.sort_values(by="sel", ascending=False).reset_index(drop=True)
            row = df.loc[(df.prefix == inp.prefix) & 
                    (df.input_text == inp.input_text),:]
            sel_row = row.index[0]
            if char == "W":
                new_row = {"prefix":inp.prefix,
                           "input_text":inp.input_text,
                           "target_text":df.loc[sel_row,"pred_text1"]}
                sel_df = sel_df.append(new_row, ignore_index=True)
            consts["sel_path"] = sel_path + "|"+ str(len(sel_df)) + "|" + str(sel_df["input_text"].nunique())
            mbeep()
            sel_df = sel_df.sort_values(by=["prefix","input_text","target_text"]).drop_duplicates()
            sel_df.to_csv(sel_path, sep="\t", index=False)
        elif char in ["h","v"] and prev_char == "x":
            _cols = ["template", "model", "prefix"]
            _types = ["l1_decoder", "l1_encoder", "cossim_decoder", "cossim_encoder"]
            canceled, col = list_values(_cols)
            folder = "/home/ahmad/share/comp/"
            if Path(folder).exists():
                shutil.rmtree(folder)
            Path(folder).mkdir(parents=True, exist_ok=True)
            files = []
            for _type in _types:
                g_list = ["template", "model", "prefix"]
                mm = main_df.groupby(g_list, as_index=False).first()
                g_list.remove(col)
                mlog.info("g_list: %s", g_list)
                g_df = mm.groupby(g_list, as_index=False)
                sel_cols = [_type, "template", "model", "prefix"]
                #_agg = {}
                #for _g in g_list:
                #    _agg[_g] ="first"
                #_agg[col] = "count"
                #df = g_df.agg(_agg)
                if True:
                    gg = 1
                    total = len(g_df)
                    for g_name, _nn in g_df:
                        img = []
                        images = []
                        for i, row in _nn.iterrows():
                            if row[_type] is None:
                                continue
                            f_path = row[_type] 
                            if not Path(f_path).is_file(): 
                                continue
                            img.append(row[_type])
                            _image = Image.open(f_path)
                            draw = ImageDraw.Draw(_image)
                            draw.text((0, 0),str(i) +" "+ row[col] ,(20,25,255),font=font)
                            draw.text((0, 70),str(i) +" "+ g_name[0],(20,25,255),font=font)
                            draw.text((0, 140),str(i) +" "+ g_name[1],(20,25,255),font=font)
                            draw.text((250, 0),str(gg) + " of " + str(total),
                                    (20,25,255),font=font)
                            images.append(_image)
                        gg += 1
                        if images:
                            if char == "h":
                                new_im = combine_x(images)
                            else:
                                new_im = combine_y(images)
                            name = _type + "_".join(g_name) + "_" + row[col]
                            pname = os.path.join(folder, name + ".png")
                            new_im.save(pname)
                            files.append({"pname":pname, "name":name})
                if files:
                    df = pd.DataFrame(files, columns=["pname","name"])
                    sel_cols = ["name"]
                else:
                    show_msg("No select")
        elif char == "x" and prev_char == "x":
            backit(df, sel_cols)
            df = sel_df
        # png files
        elif char == "l" and prev_char == "x":
            df = main_df.groupby(["l1_decoder", "template", "model", "prefix"], as_index=False).first()
            sel_cols = ["l1_decoder", "template", "model", "prefix"]
        elif char == "z" and prev_char == "x":
            fav_df = fav_df.append(df.iloc[sel_row])
            mbeep()
            fav_df.to_csv(fav_path, sep="\t", index=False)
        elif char == "Z":
            backit(df, sel_cols)
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
        elif char in "56789" and prev_char == "\\":
            cmd = "top@" + str(int(char)/10)
        elif char == "BB": 
            sel_rows = []
            for i in range(len(df)):
                sel_rows.append(i)
        elif char == "=": 
            col = sel_cols[cur_col]
            exp=df.iloc[sel_row][col]
            if col == "exp_id": col = FID
            if col == "fid":
                sel_fid = exp
            mlog.info("%s == %s", col, exp)
            df = main_df[main_df[col] == exp]
            filter_df = df
            hotkey = "gG"
        elif char  == "a": 
            col = sel_cols[cur_col]
            FID = col 
            extra["FID"] = FID
            df = filter_df
            hotkey="gG"
        elif char == "A":
            col = sel_cols[cur_col]
            FID = col 
            extra["FID"] = FID
            df = main_df
            hotkey="gG"
        elif char == "AA":
            gdf = filter_df.groupby("input_text")
            rows = []
            for group_name, df_group in gdf:
                for row_index, row in df_group.iterrows():
                    pass
            arr = ["prefix","fid","query","input_text","template"]
            canceled, col = list_values(arr)
            if not canceled:
                FID = col 
                extra["FID"] = FID
                df = filter_df
                hotkey="gG"
        elif char == "s":
            if cur_col < len(sel_cols):
                col = sel_cols[cur_col]
                if col == sort:
                    asc = not asc
                sort = col
                df = df.sort_values(by=sort, ascending=asc)
        elif char in ["c","C"] and prev_char == "c": 
            counts = {}
            for col in df:
               counts[col] = df[col].nunique()
            df = pd.DataFrame(data=[counts], columns = df.columns)
        elif char == "U": 
            if sel_col:
                df = df[sel_col].value_counts(ascending=False).reset_index()
                sel_cols = list(df.columns)
                col_widths["index"]=50
                info_cols = []
            
        elif char == "g": 
            score_col = "rouge_score"
            backit(df, sel_cols)
            group_col = "pred_text1"
            #tdf = df.groupby(['fid','prefix','input_text'],as_index=False).agg(target_text=('target_text','<br />'.join)).rename(columns={'target_text':'top_target'})
            #df = df.sort_values(score_col, ascending=False).drop_duplicates(['fid','prefix','input_text']).merge(tdf)
            df["rouge_score"] = df.groupby(['fid','prefix','input_text'])["rouge_score"].transform("max")
            df["bert_score"] = df.groupby(['fid','prefix','input_text'])["bert_score"].transform("max")
            df["hscore"] = df.groupby(['fid','prefix','input_text'])["hscore"].transform("max")
            #df["nr_score"] = df.groupby(['fid','prefix','input_text'])["nr_score"].transform("max")
            if not group_col in info_cols: info_cols.append(group_col)
            sel_cols.append("num_preds")
            extra["filter"].append("group predictions")
        elif char == " ":
            if sel_row in sel_rows:
                sel_rows.remove(sel_row)
            else:
                sel_rows.append(sel_row)
            adjust = False
        elif char == "?": 
            if "fid" in df.iloc[sel_row]:
                exp=df.iloc[sel_row]["fid"]
                sel_exp = exp
                extra["exp"] = exp
                #path = main_df.loc[main_df["fid"] == exp, "path"][0]
                #extra["path"] = path
            show_extra = True
        elif char == "z":
            sel_cols =  load_obj("sel_cols", context, [])
            info_cols = load_obj("info_cols", context, [])
        elif char == "G":
            backit(df, sel_cols)
            if FID == "input_text":
                context = "inp2"
            col = FID
            left = 0
            col = [col, "prefix"]
            sel_cols =  load_obj("sel_cols", context, [])
            info_cols = load_obj("info_cols", context, [])
            if True:
                info_cols = ["taginfo", "extra_fields"]
            if True: #col == "fid":
                sel_cols = ["trial", "tag","prefix","num_preds", "rouge_score", "steps","max_acc","best_step",  "bert_score", "st_score", "learning_rate",  "num_targets", "num_inps", "train_records", "train_records_nunique", "group_records", "wrap", "frozen", "prefixed"]

            _agg = {}
            for c in df.columns:
                if c.endswith("score"):
                    _agg[c] = "mean"
                else:
                    _agg[c] = ["first", "nunique"]
            gb = df.groupby(col)
            counts = gb.size().to_frame(name='group_records')
            df = (counts.join(gb.agg(_agg)))
            df.columns = [ '_'.join(str(i) for i in col) for col in df.columns]

            #num_targets = (df['prefix']+'_'+df['target_text']).groupby(df[col]).nunique()
            avg_len = 1 #(df.groupby(col)["pred_text1"]
                        #   .apply(lambda x: np.mean(x.str.len()).round(2)))
            ren = {
                    "target_text_nunique":"num_targets",
                    "pred_text1_nunique":"num_preds",
                    "input_text_nunique":"num_inps",
                    }
            for c in df.columns:
                if c == FID + "_first":
                    ren[c] = "exp_id"
                elif c.endswith("_mean"):
                    ren[c] = c.replace("_mean","")
                elif c.endswith("_first"):
                    ren[c] = c.replace("_first","")
            df = df.rename(columns=ren)
            #df = df.reset_index()
            df["avg_len"] = avg_len
            _infos =""
            if False:
                _sel_cols = []
                for c in sel_cols:
                    if "train_" in c:
                        mbp("sel")
                    _count = 0
                    try:
                        _count = df[c].nunique()
                        _first = df[c].iloc[0]
                    except:
                        continue
                    if _count == 1 and c != "exp_id":
                        _infos += f"{c}:{_first}  |  "
                    elif c in df and c + "_nunique" in df:
                        _gn = df[c + "_nunique"].iloc[0]
                        if _gn == 1 or c == "exp_id":
                            _sel_cols.append(c)
                    else:
                        _sel_cols.append(c)
                if _sel_cols:
                    sel_cols = _sel_cols

            extra["common"] = _infos
            df = df.sort_values(by = ["rouge_score"], ascending=False)
        elif char == "u":
            left = 0
            backit(df, sel_cols)

            s_rows = sel_rows
            if not sel_rows:
                s_rows = [sel_row]
            cond = ""
            for s_row in s_rows:
                exp=df.iloc[s_row]["exp_id"]
                cond += f"| (main_df['{FID}'] == '{exp}') "
            cond = cond.strip("|")
            filter_df = main_df[eval(cond)]
            df = filter_df.copy()
            sel_rows = []
            FID = "input_text"
            hotkey = "gG"
        elif char in ["n", "p", "t", "i"] and prev_cahr != "x":
            left = 0
            context= "comp"
            s_rows = sel_rows
            if not sel_rows:
                s_rows = [sel_row]
            sel_rows = sorted(sel_rows)
            if sel_rows:
                sel_row = sel_rows[-1]
            backit(df, sel_cols)
            on_col_list = ["pred_text1"]
            other_col = "target_text"
            if char =="i": 
                on_col_list = ["input_text"] 
                other_col = "pred_text1"
            if char =="t": 
                on_col_list = ["target_text"] 
                other_col = "pred_text1"
            on_col_list.extend(["prefix"])
            g_cols = []
            _rows = s_rows
            if char == "n":
                dfs = []
                all_rows = range(len(df))
                for r1 in all_rows:
                    for r2 in all_rows:
                        if r2 > r1:
                            _rows = [r1, r2]
                            _df, sel_exp = find_common(df, filter_df, on_col_list, _rows, FID, char)
                            dfs.append(_df)
                df = pd.concat(dfs,ignore_index=True)
                df = df.sort_values(by="int", ascending=False)
            else:
                df, sel_exp = find_common(df, filter_df, on_col_list, _rows, FID, char)
            if len(sel_rows) == 2:
                _all = len(df)
                df = df[df['pred_text1_x'].str.strip() != df['pred_text1_y'].str.strip()]
                _dif = len(df)
                _common = _all - _dif
                consts["Common"] = _common
                if "exp_name_x" in df:
                    fid_x = df.iloc[0]["exp_name_x"]
                    fid_y = df.iloc[0]["exp_name_y"]
                    df["exp_name_x"] = "|".join(list(set(fid_x.split("@")) - set(fid_y.split("@"))))
                    df["exp_name_y"] = "|".join(list(set(fid_y.split("@")) - set(fid_x.split("@"))))
            sel_cols = on_col_list
            info_cols = []
            #show_consts = False
            sel_cols.remove("prefix")
            if len(sel_rows) > 2:
                df = df.reset_index()
                _from_cols = list(df.columns) 
                sel_cols.append("target_text")
                for _col in _from_cols:
                    if _col.startswith("pred_text1"):
                        info_cols.append(_col)
            else:
                _from_cols = ["pred_text1","fid", "id", "pred_text1_x", "pred_text1_y","query_x","query_y", "query", "resp", "resp_x", "resp_y", "template", "prefix", "input_text","target_text_x", "target_text", "rouge_score", "rouge_score_x","rouge_score_y", "bert_score", "bert_score_x", "bert_score_y", "exp_name_x", "exp_name_y","sel"]
                for _col in _from_cols:
                    if (_col.startswith("id") or
                        _col.startswith("pred_text1") or 
                        _col.startswith("rouge_score") or 
                        _col.startswith("fid") or 
                        _col=="target_text" or _col=="sel" or 
                        _col.startswith("bert_score")):
                        sel_cols.append(_col)
                    elif not _col in on_col_list and not _col in info_cols:
                        info_cols.append(_col)
            info_cols.append("prefix")
            if char == "n":
                sel_cols = list(df.columns)
            sel_row = 0
            sel_rows = []
            info_cols.append("sum_fid")
            extra["short_keys"] = "m: show group" 
            info_cols_back = info_cols.copy()
            info_cols = []

        elif char == "m" and prev_char != "x":
            left = 0
            if sel_exp and on_col_list:
                backit(df, sel_cols)
                _col = on_col_list[0]
                _item=df.iloc[sel_row][_col]
                sel_row = 0
                if sel_fid:
                    df = main_df[(main_df["fid"] == sel_fid) & (main_df[FID] == sel_exp) & (main_df[_col] == _item)]
                else:
                    df = main_df[(main_df[FID] == sel_exp) & (main_df[_col] == _item)]
                sel_cols = ["fid","input_text","pred_text1","target_text","bert_score", "hscore", "rouge_score", "prefix"]
                df = df[sel_cols]
                df = df.sort_values(by="bert_score", ascending=False)
        elif char == "D" or ch == cur.KEY_SDC or char == "d":
            s_rows = sel_rows
            if FID == "fid":
                mdf = main_df.groupby("fid", as_index=False).first()
                mdf = mdf.copy()
                _sels = df["exp_id"]
                for s_row, row in mdf.iterrows():
                    exp=row["fid"]
                    if char == "d":
                        cond = main_df['fid'].isin(_sels) 
                    else:
                        cond = ~main_df['fid'].isin(_sels) 
                    tdf = main_df[cond]
                    if  ch == cur.KEY_SDC:
                        spath = row["path"]
                        os.remove(spath)
                    main_df = main_df.drop(main_df[cond].index)
                df = main_df
                filter_df = main_df
                sel_rows = []
                hotkey = "gG"
        elif char == "D" and prev_char == "x":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                del main_df[col]
                char = "SS"
                if col in df:
                    del df[col]
        elif char == "o" and prev_char == "x":
            if "pname" in df:
                pname = df.iloc[sel_row]["pname"]
            elif "l1_encoder" in df:
                if not sel_rows: sel_rows = [sel_row]
                sel_rows = sorted(sel_rows)
                pnames = []
                for s_row in sel_rows:
                    pname1 = df.iloc[s_row]["l1_encoder"]
                    pname2 = df.iloc[s_row]["l1_decoder"]
                    pname3 = df.iloc[s_row]["cossim_encoder"]
                    pname4 = df.iloc[s_row]["cossim_decoder"]
                    images = [Image.open(_f) for _f in [pname1, pname2,pname3, pname4]]
                    new_im = combine_y(images)
                    name = "temp_" + str(s_row) 
                    folder = os.path.join(base_dir, "images")
                    Path(folder).mkdir(exist_ok=True, parents=True)
                    pname = os.path.join(folder, name + ".png")
                    draw = ImageDraw.Draw(new_im)
                    draw.text((0, 0), str(s_row) + "  " + df.iloc[s_row]["template"] +  
                                     " " + df.iloc[s_row]["model"] ,(20,25,255),font=font)
                    new_im.save(pname)
                    pnames.append(pname)
                if len(pnames) == 1:
                    pname = pnames[0]
                    sel_rows = []
                else:
                    images = [Image.open(_f) for _f in pnames]
                    new_im = combine_x(images)
                    name = "temp" 
                    folder = os.path.join(base_dir, "images")
                    Path(folder).mkdir(exist_ok=True, parents=True)
                    pname = os.path.join(folder, name + ".png")
                    new_im.save(pname)
            if "ahmad" in home:
                subprocess.run(["eog", pname])
        elif char in ["o","O"] and prev_char=="x":
            files = [Path(f).stem for f in glob(base_dir+"/*.tsv")]
            for i,f in enumerate(files):
                if f in open_dfnames:
                    files[i] = "** " + f

            canceled, _file = list_values(files)
            if not canceled:
                open_dfnames.append(_file)
                _file = os.path.join(base_dir, _file + ".tsv")
                extra["files"] = open_dfnames
                new_df = pd.read_table(_file)
                if char == "o":
                    df = pd.concat([df, new_df], ignore_index=True)
                else:
                    main_df = pd.concat([main_df, new_df], ignore_index=True)
        elif char == "t" and prev_char=="x":
            cols = get_cols(df,5)
            if cols:
                tdf = df[cols].round(2)
                tdf = tdf.pivot(index=cols[0], columns=cols[1], values =cols[2]) 
                fname = rowinput("Table name:", "table_")
                if fname:
                    if char == "t":
                        tname = os.path.join(base_dir, "plots", fname + ".png")
                        wrate = [col_widths[c] for c in cols]
                        tax = render_mpl_table(tdf, wrate = wrate, col_width=4.0)
                        fig = tax.get_figure()
                        fig.savefig(tname)
                    else:
                        latex = tdf.to_latex(index=False)
                        tname = os.path.join(base_dir, "latex", fname + ".tex")
                        with open(tname, "w") as f:
                            f.write(latex)

        elif char == "P":
            cols = get_cols(df,2)
            if cols:
                df = df.sort_values(cols[1])
                x = cols[0]
                y = cols[1]
                #ax = df.plot.scatter(ax=ax, x=x, y=y)
                ax = sns.regplot(df[x],df[y])
        elif is_enter(ch) or char in ["f", "F"]:
            backit(df, sel_cols)
            if is_enter(ch): char = "F"
            col = sel_cols[cur_col]
            if col == "exp_id": col = FID
            if char == "f":
                canceled, col, val = list_df_values(main_df, col, get_val=True)
            else:
                canceled, col, val = list_df_values(filter_df, col, get_val=True)
            if not canceled:
               if char == "F" and prev_char == "x":
                    cond = get_cond(filter_df, col, num=15)
               else:
                    if not canceled:
                        if type(val) == str:
                            cond = f"df['{col}'] == '{val}'"
                        else:
                            cond = f"df['{col}'] == {val}"
               if cond:
                   mlog.info("cond %s, ", cond)
                   if char == "f":
                       df = main_df
                   else:
                       df = filter_df
                   df = df[eval(cond)]
                   #df = df.reset_index()
                   filter_df = df
                   if not "filter" in extra:
                        extra["filter"] = []
                   extra["filter"].append(cond)
                   sel_row = 0
                   if char == "F" or char == "f":
                       hotkey = "gG"
        if char in ["y","Y"]:
            #yyyyyyyy
           cols = get_cols(df, 2)
           backit(df, sel_cols)
           if cols:
               gcol = cols[0]
               y_col = cols[1]
               if char == "Y":
                   cond = get_cond(df, gcol, 10)
                   df = df[eval(cond)]
               gi = 0 
               name = ""
               for key, grp in df.groupby([gcol]):
                     ax = grp.sort_values('steps').plot.line(ax=ax,linestyle="--",marker="o",  x='steps', y=y_col, label=key, color=colors[gi])
                     gi += 1
                     if gi > len(colors) - 1: gi = 0
                     name += key + "_"
               ax.set_xticks(df["steps"].unique())
               ax.set_title(name)
               if not "filter" in extra:
                   extra["filter"] = []
               extra["filter"].append("group by " + name)
               char = "H"
        if char == "H":
            name = ax.get_title()
            pname = rowinput("Plot name:", name[:30])
            if pname:
                folder = ""
                if "/" in pname:
                    folder, pname = pname.split("/")
                ax.set_title(pname)
                if folder:
                    folder = os.path.join(base_dir, "plots", folder)
                else:
                    folder = os.path.join(base_dir, "plots")
                Path(folder).mkdir(exist_ok=True, parents=True)
                pname = pname.replace(" ", "_")
                pname = os.path.join(folder, now + "_" + pname +  ".png")
                fig = ax.get_figure()
                fig.savefig(pname)
                ax = None
                if "ahmad" in home:
                    subprocess.run(["eog", pname])

        elif char == "R" and prev_char == "x":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                new_name = rowinput(f"Rename {col}:")
                main_df = main_df.rename(columns={col:new_name})
                char = "SS"
                if col in df:
                    df = df.rename(columns={col:new_name})



        elif char in ["d"] and prev_char == "x":
            canceled, col, val = list_df_values(main_df)
            if not canceled:
                main_df = main_df.drop(main_df[main_df[col] == val].index)
                char = "SS"
                info_cols = []
                if col in df:
                    df = df.drop(df[df[col] == val].index)
        elif ch == cur.KEY_DC:
            col = sel_cols[cur_col]
            sel_cols.remove(col)
            save_obj(sel_cols, "sel_cols", context)
        elif ch == cur.KEY_SDC and prev_char == 'x':
            col = sel_cols[0]
            val = sel_dict[col]
            cmd = rowinput("Are you sure you want to delete {} == {} ".format(col,val))
            if cmd == "y":
                main_df = main_df.drop(main_df[main_df[col] == val].index)
                char = "SS"
                info_cols = []
                if col in df:
                    df = df.drop(df[df[col] == val].index)
        elif char == "M" and prev_char == "x":
            info_cols = []
            for col in df.columns:
                info_cols.append(col)
        elif char == "m" and prev_char == "x":
            info_cols = []
            sel_cols = []
            cond = get_cond(df, "model", 2)
            df = main_df[eval(cond)]
            if df.duplicated(['qid','model']).any():
                show_err("There is duplicated rows for qid and model")
                char = "r"
            else:
                df = df.set_index(['qid','model'])[['pred_text1', 'input_text','prefix']].unstack()
                df.columns = list(map("_".join, df.columns))
        elif is_enter(ch) and prev_char == "x":
            col = sel_cols[0]
            val = sel_dict[col]
            if not "filter" in extra:
                extra["filter"] = []
            extra["filter"].append("{} == {}".format(col,val))
            df = filter_df[filter_df[col] == val]
            df = df.reset_index()
            if char == "F":
                sel_cols = order(sel_cols, [col])
            sel_row = 0
            filter_df = df
        elif char == "w" and prev_cahr == "x":
            sel_rows = []
            adjust = True
            tdf = main_df[main_df['fid'] == sel_exp]
            spath = tdf.iloc[0]["path"]
            tdf.to_csv(spath, sep="\t", index=False)
        elif char == "/" and prev_char == "x":
            old_search = search
            search = rowinput("/", search)
            if search == old_search:
                si += 1
            else:
                si = 0
            mask = np.column_stack([df[col].astype(str).str.contains(search, na=False) for col in df])
            si = min(si, len(mask) - 1)
            sel_row = df.loc[mask.any(axis=1)].index[si]
        elif char == ":":
            cmd = rowinput() #default=prev_cmd)
        elif char == "q":
            cmd = rowinput("Are you sure you want to exit? (y/n)")
            if cmd == "y":
                ch = ord("q")
                save_df(df)
            else:
                ch = 0
        if cmd.startswith("w="):
            _,val = cmd.split("=")[1]
            col = sel_cols[cur_col]
            col_widths[col] = int(val)
            adjust = False
        if cmd == "report":
            doc_dir = os.path.join(home, "Documents/Paper1/icml-kr")
            with open(f"{doc_dir}/report.tex.temp", "r") as f:
                report = f.read()
            table_dir = os.path.join(doc_dir, "table")
            all_steps = df['steps'].unique()
            for rel in df['prefix'].unique(): 
                with open(f"{table_dir}/table.txt", "r") as f:
                    table_cont = f.read()
                for samp in all_steps:
                    cont = table_cont
                    table_name = f"{rel}_{samp}.txt"
                    table_file = f"{table_dir}/{table_name}"
                    _input = f"table/{table_name}" 
                    out = open(table_file, "w")
                    for met in df["template"].unique():
                        for mod in ["t5-v1", "t5-lmb", "t5-base"]:
                            for sc in ["rouge_score", "bert_score", "hscore", "num_preds"]:
                                cond = ((df['prefix'] == rel) &
                                        (df["template"] == met) &
                                        (df["steps"] == samp) &
                                        (df["model"] == mod))
                                val = df.loc[cond, sc].mean()
                                val = round(val,2)
                                if sc != "num_preds":
                                    val = "{:.2f}".format(val)
                                else:
                                    try:
                                       val = str(int(val))
                                    except:
                                       val = "NA"
                                cont = cont.replace("@" + met + "@" + mod + "@" + sc, val)
                    out.write(cont)
                    out.close()
                    lable = "results:" + rel + "_" + samp
                    caption = f"{rel} for {samp}"
                    table = """
                        \\begin{{table*}}
                            \centering
                            \label{{{}}}
                            \caption{{{}}}
                            \\begin{{tabular}}{{|c| c|c|c| c | c |}}
                            \input{{{}}}
                            \end{{tabular}}
                        \end{{table*}}
                        """
                    table = table.format(lable, caption, _input)
                    report = report.replace("mytable", table +"\n\n" + "mytable")
            with open(f"{doc_dir}/report.tex", "w") as f:
                f.write(report)

        if cmd == "fix_types":
            for col in ["target_text", "pred_text1"]: 
                main_df[col] = main_df[col].astype(str)
            for col in ["steps", "epochs", "val_steps"]: 
                main_df[col] = main_df[col].astype(int)
            char = "SS"
        if cmd == "clean":
            main_df = main_df.replace(r'\n',' ', regex=True)
            char = "SS"
        if cmd == "fix_template":
            main_df.loc[(df["template"] == "unsup-tokens") & 
                    (main_df["wrap"] == "wrapped-lstm"), "template"] = "unsup-tokens-wrap"
            main_df.loc[(main_df["template"] == "sup-tokens") & 
                    (main_df["wrap"] == "wrapped-lstm"), "template"] = "sup-tokens-wrap"
        
        if cmd == "repall":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                _a = rowinput("from")
                _b = rowinput("to")
                main_df[col] = main_df[col].str.replace(_a,_b)
                char = "SS"
        if cmd == "rep" or cmd == "rep@":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                vals = df[col].unique()
                d = {}
                for v in vals:
                    rep = rowinput(str(v) + "=" ,v)
                    if not rep:
                        break
                    if type(v) == int:
                        d[v] = int(rep)
                    else:
                        d[v] = rep
                if rowinput("Apply?") == "y":
                    if "@" in cmd:
                        df = df.replace(d)
                    else:
                        df = df.replace(d)
                        main_df = main_df.replace(d)
                        char = "SS"
        if cmd in ["set", "set@", "add", "add@", "setcond"]:
            if "add" in cmd:
                col = rowinput("New col name:")
            else:
                canceled, col,val = list_df_values(main_df, get_val=False)
            cond = ""
            if "cond" in cmd:
                cond = get_cond(df, num=5, op="&")
            if not canceled:
                if cond:
                    val = rowinput(f"Set {col} under {cond} to:")
                else:
                    val = rowinput("Set " + col + " to:")
                if val:
                    if cond:
                        if "@" in cmd:
                            df.loc[eval(cond), col] = val
                        else:
                            main_df[eval(cond), col] =val
                            char = "SS"
                    else:
                        if "@" in cmd:
                            df[col] = val
                        else:
                            main_df[col] =val
                            char = "SS"
        if "==" in cmd:
            col, val = cmd.split("==")
            df = df[df[col] == val]
        elif "top@" in cmd:
            backit(df, sel_cols)
            tresh = float(cmd.split("@")[1])
            df = df[df["bert_score"] > tresh]
            df = df[["prefix","input_text","target_text", "pred_text1"]] 
        if cmd == "cp" or cmd == "cp@":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                copy = rowinput("Copy " + col + " to:", col)
                if copy:
                    if "@" in cmd:
                        df[copy] = df[col]
                    else:
                        main_df[copy] = main_df[col]
                        char = "SS"
        if cmd.isnumeric():
            sel_row = int(cmd)
        elif cmd == "q":
            save_df(df)
            ch = ord("q")
        elif not char in ["q", "S","r"]:
            pass
            #mbeep()
        if char in ["S", "}"]:
            _name = "main_df" if char == "S" else "df"
            _dfname = dfname
            if dfname == "merged":
                _dfname = "test"
            cmd, _ = minput(cmd_win, 0, 1, f"File Name for {_name} (without extension)=", default=_dfname, all_chars=True)
            cmd = cmd.split(".")[0]
            if cmd != "<ESC>":
                if char == "}":
                    df.to_csv(os.path.join(base_dir, cmd+".tsv"), sep="\t", index=False)
                else:
                    dfname = cmd
                    char = "SS"
        if char == "SS":
                df = main_df[["prefix","input_text","target_text"]]
                df = df.groupby(['input_text','prefix','target_text'],as_index=False).first()

                save_path = os.path.join(base_dir, dfname+".tsv")
                sel_cols = ["prefix", "input_text", "target_text"]
                Path(base_dir).mkdir(parents = True, exist_ok=True)
                df.to_csv(save_path, sep="\t", index=False)

                save_obj(dfname, "dfname", dfname)
        if char == "r" and prev_char != "x":
            filter_df = main_df
            df = filter_df
            FID = "fid" 
            sel_cols = []
            save_obj([], "sel_cols", context)
            save_obj([], "info_cols", context)
            hotkey = "gG"
        if char == "r" and prev_char == "x":
            df = main_df
            sel_cols = list(df.columns)
            save_obj(sel_cols,"sel_cols",dfname)
            extra["filter"] = []
            info_cols = []
        if char == "b" and back:
            if back:
                df = back.pop()
                sel_cols = sels.pop() 
                sel_row = back_rows.pop()
                left = 0
            else:
                mbeep()
            if extra["filter"]:
                extra["filter"].pop()

def render_mpl_table(data, wrate, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        mlog.info("Size %s", size)
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    mpl_table.auto_set_column_width(col=list(range(len(data.columns)))) # Provide integer list of columns to adjust

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax
def get_cond(df, for_col = "", num = 1, op="|"):
    canceled = False
    sels = []
    cond = ""
    while not canceled and len(sels) < num:
        canceled, col, val = list_df_values(df, col=for_col, get_val=True,sels=sels)
        if not canceled:
            cond += f"{op} (df['{col}'] == '{val}') "
            sels.append(val)
    cond = cond.strip(op)
    return cond

def get_cols(df, num = 1):
    canceled = False
    sels = []
    while not canceled and len(sels) < num:
        canceled, col,_ = list_df_values(df, get_val=False, sels = sels)
        if not canceled:
            sels.append(col)
    return sels

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

def order(sel_cols, cols, pos=0):
    z = [item for item in sel_cols if item not in cols] 
    z[pos:pos] = cols
    save_obj(z, "sel_cols",dfname)
    return z


                
def change_info(infos):
    info_bar.erase()
    h,w = info_bar.getmaxyx()
    w = 80
    lnum = 0
    for msg in infos:
        lines = textwrap.wrap(msg, width=w, placeholder=".")
        for line in lines: 
            mprint(str(line), info_bar, color=HL_COLOR)
            lnum += 1
    rows,cols = std.getmaxyx()
    info_bar.refresh(0,0, rows -lnum - 1,0, rows-1, cols - 2)
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
dfpath = ""
dftype = "full"
base_dir = os.path.join(home, "mt5-comet", "comet", "data", "atomic2020" , "sel")
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
    #if not dfname:
    #    fname = load_obj("dfname","","")
    #    dfname = fname + ".tsv"
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
                        root_file = os.path.join(root,_file)
                        if dftype in _file and all(s.strip() in root_file for s in dfname.split("+")):
                            files.append(root_file)
        mlog.info("files: %s",files)
        if not files:
            print("No file was selected")
            return
        dfs = []
        for ii, f in enumerate(files):
            mlog.info(f)
            print(f)
            print("==================")
            if f.endswith(".tsv"):
                df = pd.read_table(f, low_memory=False)
            elif f.endswith(".json"):
                df = load_results(f)
            force_fid = False
            sfid = file_id.split("@")
            fid = sfid[0]
            if len(sfid) > 1:
                force_fid = sfid[1] == "force"
            if True: #force_fid:
                df["path"] = f
                df["fid"] = ii
                _dir = str(Path(f).parent)
                _pp = _dir + "/*.png"
                png_files = glob(_pp)
                if not png_files:
                    _pp = str(Path(_dir).parent) + "/hf*/*.png"
                    png_files = glob(_pp)
                for i,png in enumerate(png_files):
                    key = Path(png).stem
                    if not key in df:
                       df[key] = png
                if fid == "parent":
                    _ff = "@".join(f.split("/")[5:]) 
                    df["exp_name"] = _ff #.replace("=","+").replace("_","+")
                elif fid == "name":
                    df["exp_name"] =  "_" + Path(f).stem
                else:
                    df["exp_name"] =  "_" + df[fid]
            dfs.append(df)
        if len(dfs) > 1:
            df = pd.concat(dfs, ignore_index=True)
            dfname = "merged"
        if files:
            show_df(df)
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
@click.option(
    "--ftype",
    "-ft",
    default="full",
    type=str,
    help=""
)
@click.option(
    "--dpy",
    "-dpy",
    is_flag=True,
    help=""
)
def main(fname, path, fid, ftype, dpy):
    if dpy:
        port = 5678
        debugpy.listen(('0.0.0.0', int(port)))
        print("Waiting for client at run...port:", port)
        debugpy.wait_for_client()  # blocks execution until client is attached
    global dfname,dfpath,file_id,dftype
    file_id = fid
    if not fname:
        fname = [dftype]
    if fname != "last":
        dfname = fname 
        dfpath = path
    dftype= ftype
    set_app("showdf")
    wrapper(start)

if __name__ == "__main__":
    main()
