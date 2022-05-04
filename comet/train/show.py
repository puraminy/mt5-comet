import curses as cur
import subprocess
from functools import reduce
import matplotlib.pyplot as plt
from curses import wrapper
import click
import numpy as np
from glob import glob
import six
import os
import re
import seaborn as sns
from pathlib import Path
import pandas as pd
from comet.mycur.util import *
from mylogs import * 
import json
from comet.utils.myutils import *
file_id = "name"
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
    df.to_csv(path.replace("json", "tsv"), sep="\t", index = False)
    return df

def find_common(df, main_df, on_col_list, s_rows, FID, char):
    dfs_items = [] 
    dfs = []
    ii = 0
    dfs_val = {}
    for s_row in s_rows:
        exp=df.iloc[s_row]["exp_id"]
        dfs_val["exp" + str(ii)] = exp
        tdf = main_df[main_df[FID] == exp]
        tdf = tdf[["pred_text1", "bert_score","qid", "method", "rouge_score", "fid","prefix", "input_text","target_text"]]
        tdf = tdf.sort_values(by="rouge_score")
        tdf = tdf.groupby(on_col_list).agg({"qid":"first","input_text":"first","target_text":"first", "method":"first", "rouge_score":"mean","prefix":"first","pred_text1":"first", "fid":"count"}).reset_index(drop=True)
        tdf = tdf.sort_values(by="fid", ascending=False)
        for on_col in on_col_list:
            tdf[on_col] = tdf[on_col].astype(str).str.strip()
        dfs.append(tdf) #.copy())
        ii += 1
    if ii > 1:
        intersect = reduce(lambda  left,right: pd.merge(left,right,on=on_col_list,
                                    how='inner'), dfs)
        if char == "a":
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
        text_win.refresh(0, left, 0, 0, ROWS-1, COLS-2)
    def fill_text_win(rows):
        text_win.erase()
        for row in rows:
            mprint(row, text_win)
        refresh()



    if not col_widths:
        col_widths = {"qid":5, "model":30, "pred_text1":30, "epochs":30, "date":30, "rouge_score":7, "bert_score":7, "input_text":50}

    df['id']=df.index
    df = df.reset_index(drop=True)
    if "input_text" in df:
        df['input_text'] = df['input_text'].str.replace('##','')
        df['input_text'] = df['input_text'].str.split('>>').str[0]
        df['input_text'] = df['input_text'].str.strip()
    main_df = df
    edit_col = ""
    count_col = ""
    consts = {"filter":[]}
    save_obj(dfname, "dfname", "")
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

    back = []
    filter_df = main_df
    FID = "method"
    if "pred_text1" in df:
        df["num_preds"] = df.groupby([FID])[['pred_text1']].transform('nunique')
        df["num_query"] = df.groupby([FID])[['qid']].transform('nunique')
        br_col = df.loc[: , "bert_score":"rouge_score"]
        df['br_score'] = br_col.mean(axis=1)
        df['nr_score'] = df['rouge_score']
        df['nr_score'] = np.where((df['bert_score'] > 0.4) & (df['nr_score'] < 0.1), df['bert_score'], df['rouge_score'])


    #wwwwwwwwww
    colors = ['blue','orange','green', 'red', 'purple', 'brown', 'pink','gray','olive','cyan']
    ax = None
    seq = ""
    search = ""
    open_dfnames = [dfname]
    if not "blank" in df:
        df["blank"] = "blank"
    prev_cahr = ""
    hotkey = ""
    sel_exp = ""
    back_row = 0
    sel_rows = []
    while ch != ord("q"):
        text_win.erase()
        left = min(left, max_col  - width)
        left = max(left, 0)
        sel_row = min(sel_row, len(df) - 1)
        sel_row = max(sel_row, 0)
        if not hotkey:
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
               _sels = []
               _infs = []
               for sel_col in sel_cols + info_cols:
                   if not sel_col in row or sel_col in _sels:
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
                   _color = TEXT_COLOR
                   _w = col_widths[sel_col] if sel_col in col_widths else width
                   if sel_col in sel_cols:
                       text += "{:<{x}}".format(content, x= _w)
                       _sels.append(sel_col)

               if ii in sel_rows:
                   _color = HL_COLOR
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
        infos.append(f"hotkey:{hotkey}")
        consts["len"] = str(len(df))
        for key,val in consts.items():
            if type(val) == list:
                val = "-".join(val)
            infos.append("{:<5}:{}".format(key,val))
        change_info(infos)

        prev_char = chr(ch)
        if hotkey == "":
            ch = std.getch()
        else:
            ch, hotkey = ord(hotkey[0]), hotkey[1:]

        char = chr(ch)
        seq += char
        vals = []
        get_cmd = False
        if ch == LEFT:
            left -= width
        if ch == RIGHT:
            left += width
        if ch == DOWN or char == "N":
            sel_row += 1
        elif ch == UP or char == "B":
            sel_row -= 1
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
        elif char == "l":
            seq = ""
        elif char in list("01234"):
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
        elif char in ["%"]:
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
            back.append(df)
            df = sel_df
        elif char == "z":
            fav_df = fav_df.append(df.iloc[sel_row])
            mbeep()
            fav_df.to_csv(fav_path, sep="\t", index=False)
        elif char == "Z":
            back.append(df)
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
        elif char in "56789":
            ii = int(char) - 5
            arr = ["qid","fid","method"]
            if ii < len(arr):
                FID = arr[ii] 
                df = main_df
                df["num_preds"] = df.groupby([FID])['pred_text1'].transform('nunique')
                df["num_query"] = df.groupby([FID])['qid'].transform('nunique')
                hotkey="gG"
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
            
        elif char == "g": 
            score_col = "rouge_score"
            back.append(df)
            group_col = "pred_text1"
            df = df.sort_values(score_col, ascending=False).\
                 drop_duplicates(['fid','prefix','input_text']).\
                    rename(columns={group_col:'top_target'}).\
                      merge(df.groupby(['fid','prefix','input_text'],as_index=False)[group_col].agg('<br />'.join))
            if not group_col in info_cols: info_cols.append(group_col)
            consts["filter"].append("group predictions")
        elif char == " ":
            if sel_row in sel_rows:
                sel_rows.remove(sel_row)
            else:
                sel_rows.append(sel_row)
        elif char == "?": 
            exp=df.iloc[sel_row]["exp_id"]
            sel_exp = exp
            consts["exp"] = exp
            path = main_df.loc[main_df["fid"] == exp, "path"][0]
            consts["path"] = path
        elif char in ["G", "A"]:
            back.append(df)
            if char ==  "A":
                canceled, col, _ = list_df_values(df, get_val=False)
            elif char == "G":
                canceled, col = False, FID
            if not canceled:
               sel_cols = ["exp_id", "num_preds", "num_query", "rouge_score", "bert_score", "br_score","nr_score", "steps", "method","model", "wrap", "frozen", "prefixed"]
               df = (df.groupby(col).agg({"num_preds":"first", "num_query":"first", "rouge_score":"mean","bert_score":"mean","br_score": "mean", "nr_score":"mean", "method":"first","model":"first", "wrap":"first", col:"first", "steps":"first", "frozen":"first", "prefixed":"first"})
                 .rename(columns={col:'exp_id'})
                 .sort_values(by = ["rouge_score"], ascending=False)
                    )
               #df = df.reset_index()
               consts["filter"].append("group experiments")
               df["exp_id"] = df["exp_id"].astype(str)
               col_widths["exp_id"] = 2 + df.exp_id.str.len().max()
        elif char == "n":
            hotkey = "bNh"
        elif char == "u":
            left = 0
            back.append(df)
            back_row = sel_row

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
            FID = "qid"
            hotkey = "gG"
        elif char in ["a", "h", "T"]:
            left = 0
            back.append(df)
            back_row = sel_row

            s_rows = sel_rows
            if not sel_rows:
                s_rows = [sel_row]
            on_col_list = ["pred_text1"]
            other_col = "target_text"
            if char =="T": 
                on_col_list = ["input_text"] 
                other_col = "pred_text1"
            on_col_list.extend(["prefix"])
            g_cols = []
            _rows = s_rows
            if char == "a":
                dfs = []
                all_rows = range(len(df))
                for r1 in all_rows:
                    for r2 in all_rows:
                        if r2 > r1:
                            _rows = [r1, r2]
                            _df = find_common(df, filter_df, on_col_list, _rows, FID, char)
                            dfs.append(_df)
                df = pd.concat(dfs,ignore_index=True)
                df = df.sort_values(by="int", ascending=False)
            else:
                df = find_common(df, filter_df, on_col_list, _rows, FID, char)
            sel_cols = on_col_list
            for _col in df.columns:
                if (_col.startswith("fid") or
                    _col.startswith("pred_text1") or 
                    _col.startswith("target_text") or 
                    _col.startswith("input_text") or 
                    _col.startswith("method") or 
                    _col.startswith("prefix")):
                    if (_col.startswith("fid") or
                        _col.startswith("pred_text1") or 
                        _col.startswith("method")):
                        sel_cols.append(_col)
                    elif not _col in on_col_list and not _col in info_cols:
                        info_cols.append(_col)
                    g_cols.append(_col)
                    col_widths[_col] = 20
                    if (_col.startswith("pred_text1") or
                        _col.startswith("target_text")):
                        col_widths[_col] = 30
                    if _col.startswith("fid"):
                        col_widths[_col] = 10
            if char == "a":
                sel_cols = list(df.columns)
            else:
                if len(s_rows) > 1:
                    sel_cols.extend([other_col + "_x", other_col + "_y", "method_x", "method_y", "fid_x", "fid_y" ])
                #    info_cols = ["prefix", "target_text", "input_text", "target_text_x", "target_text_y", "pred_text1_x", "pred_text1_y"]
                else:
                    sel_cols.extend([other_col, "method"])
                #    info_cols = ["prefix", "target_text", "input_text", "pred_text1"]

            sel_row = 0
            sel_rows = []

        elif char == "H":
            left = 0
            if sel_exp:
                back.append(df)
                pred=df.iloc[sel_row]["pred_text1"]
                sel_row = 0
                consts["pred"] =pred 
                df = main_df[(main_df["method"] == sel_exp) & (main_df["pred_text1"] == pred)]
                df = df[["pred_text1","target_text","rouge_score","input_text", "prefix"]]
                df = df.sort_values(by="rouge_score", ascending=False)
        elif char == "D":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                del main_df[col]
                char = "SS"
                if col in df:
                    del df[col]
        elif char in ["o","O"]:
            files = [Path(f).stem for f in glob(base_dir+"/*.tsv")]
            for i,f in enumerate(files):
                if f in open_dfnames:
                    files[i] = "** " + f

            canceled, _file = list_values(files)
            if not canceled:
                open_dfnames.append(_file)
                _file = os.path.join(base_dir, _file + ".tsv")
                consts["files"] = open_dfnames
                new_df = pd.read_table(_file)
                if char == "o":
                    df = pd.concat([df, new_df], ignore_index=True)
                else:
                    main_df = pd.concat([main_df, new_df], ignore_index=True)
        elif char in ["t", "x"]:
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

        elif char == "p":
            cols = get_cols(df,2)
            if cols:
                df = df.sort_values(cols[1])
                x = cols[0]
                y = cols[1]
                #ax = df.plot.scatter(ax=ax, x=x, y=y)
                ax = sns.regplot(df[x],df[y])

        elif char in ["f", "F"]:
            back.append(df)
            canceled, col, val = list_df_values(df, get_val=True)
            if not canceled:
               if char == "F":
                    cond = get_cond(df, col, num=15)
               else:
                    if not canceled:
                        if type(val) == str:
                            cond = f"df['{col}'] == '{val}'"
                        else:
                            cond = f"df['{col}'] == {val}"
               if cond:
                   mlog.info("cond %s, ", cond)
                   df = df[eval(cond)]
                   df = df.reset_index()
                   if not "filter" in consts:
                        consts["filter"] = []
                   consts["filter"].append(cond)
                   sel_row = 0
        if char in ["y","Y"]:
            #yyyyyyyy
           cols = get_cols(df, 2)
           back.append(df)
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
               if not "filter" in consts:
                   consts["filter"] = []
               consts["filter"].append("group by " + name)
               char = "P"
        if char == "P":
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
        elif char == "R":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                new_name = rowinput(f"Rename {col}:")
                main_df = main_df.rename(columns={col:new_name})
                char = "SS"
                if col in df:
                    df = df.rename(columns={col:new_name})



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
        elif char == "M":
            info_cols = []
            for col in df.columns:
                info_cols.append(col)
        elif char == "m":
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
                for s in sel_cols:
                    col_widths[s] = 35
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
                cmd = rowinput(":width=", str(col_widths[col]))
                if cmd.isnumeric():
                    col_widths[col] = int(cmd)
                    save_obj(col_widths, "widths", "")
        elif char == "/":
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
            cmd = rowinput()
            if cmd == "fix_types":
                for col in ["target_text", "pred_text1"]: 
                    main_df[col] = main_df[col].astype(str)
                for col in ["steps", "epochs", "val_steps"]: 
                    main_df[col] = main_df[col].astype(int)
                char = "SS"
            if cmd == "clean":
                main_df = main_df.replace(r'\n',' ', regex=True)
                char = "SS"
            if cmd == "fix_method":
                main_df.loc[(df["method"] == "unsup-tokens") & 
                        (main_df["wrap"] == "wrapped-lstm"), "method"] = "unsup-tokens-wrap"
                main_df.loc[(main_df["method"] == "sup-tokens") & 
                        (main_df["wrap"] == "wrapped-lstm"), "method"] = "sup-tokens-wrap"
            
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
        if char in ["S", "V"]:
            cmd, _ = minput(cmd_win, 0, 1, "File Name (without extension)=", default=dfname, all_chars=True)
            cmd = cmd.split(".")[0]
            if cmd != "<ESC>":
                if char == "V":
                    df.to_csv(os.path.join(base_dir, cmd+".tsv"), sep="\t", index=False)
                else:
                    dfname = cmd
                    char = "SS"
        if char == "SS":
                main_df.to_csv(os.path.join(base_dir, dfname+".tsv"), sep="\t", index=False)
                save_obj(dfname, "dfname", dfname)
        if char == "r":
            df = main_df
            sel_cols = list(df.columns)
            save_obj(sel_cols,"sel_cols",dfname)
            consts["filter"] = []
            info_cols = []
        if char == "b" and back:
            if back:
                df = back.pop()
                sel_cols = list(df.columns)
                sel_row = back_row
            else:
                mbeep()
            if consts["filter"]:
                consts["filter"].pop()

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
    if not dfname:
        fname = load_obj("dfname","","")
        dfname = fname + ".tsv"
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
                        if all(s.strip() in root_file for s in dfname.split("+")):
                            files.append(root_file)
        mlog.info("files: %s",files)
        dfs = []
        for ii, f in enumerate(files):
            mlog.info(f)
            if f.endswith(".tsv"):
                df = pd.read_table(f, low_memory=False)
            elif f.endswith(".json"):
                df = load_results(f)
            force_fid = False
            sfid = file_id.split("@")
            fid = sfid[0]
            if len(sfid) > 1:
                force_fid = sfid[1] == "force"
            if not "fid" in df or force_fid:
                df["path"] = f
                if fid == "parent":
                    df["fid"] = str(ii) + "_" + "_".join(f.split("/")[2:]) 
                elif fid == "name":
                    df["fid"] = str(ii) + "_" + Path(f).stem
                else:
                    df["fid"] = str(ii) + "_" + df[fid]
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
def main(fname, path, fid):
    global dfname,dfpath,file_id
    file_id = fid
    if fname != "last":
        dfname = fname 
        dfpath = path
    set_app("showdf")
    wrapper(start)

if __name__ == "__main__":
    main()
