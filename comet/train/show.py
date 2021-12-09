import curses as cur
from curses import wrapper
import click
import os
from pathlib import Path
import pandas as pd
from nodcast.util.util import *
from comet.train.common import *

def load_results(fname):
    with open(os.path.join(base_dir, fname + ".json"), "r") as f:
        data = json.load(f)
    sd = superitems(data)
    if fname == "results":
        main_df = pd.DataFrame(sd, columns=["exp","model","lang", "method","wrap","frozen","epochs","stype", "date", "dir", "score"])
    elif fname == "full_results":
        main_df = pd.DataFrame(sd, columns=["qid","exp","model","lang", "method","wrap","frozen","epochs","date", "field", "text"])

    out = f"{fname}.tsv"
    df = main_df.pivot(index=list(main_df.columns[~main_df.columns.isin(['field', 'text'])]), columns='field').reset_index()

    #df.columns = list(map("".join, df.columns))
    df.columns = [('_'.join(str(s).strip() for s in col if s)).replace("text_","a_") for col in df.columns]
    df.to_csv(os.path.join(resPath, out), sep="\t", index = False)
    return df

def show_df(df):
    global dfname
    cmd = ""
    sel_row = 0
    ROWS, COLS = std.getmaxyx()
    ch = 1
    sel_cols = load_obj("sel_cols", dfname)
    if not sel_cols:
        sel_cols = list(df.columns)
    sel_row = 0
    left = 0
    max_row, max_col= text_win.getmaxyx()
    width = 15
    cond = ""
    main_df = df
    sort = ""
    asc = False
    info_cols = load_obj("info_cols", dfname, []) 
    sel_vals = []
    stats = []
    col_widths = load_obj("widths", dfname)
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
            text += "{:<{x}}".format(f"{i}) {sel_col}" , x=_w) 
        mprint(text, text_win) 
        ii = 0
        top_margin = min(len(df), 5)
        #fffff
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
        infos = []
        for c in info_cols:
            if not c in df:
                continue
            _info = df.at[sel_row, c]
            if "score" in c:
                mean = df[c].mean()
                _info = f"Mean {c}:" + "{:.2f}".format(mean)
            infos.append(_info)
        infos.append("-------------------------")
        stats = ["> " + str(len(df))]
        for stat in stats:
            infos.append(stat)
        change_info(infos)

        if store_back:
            back = {"df":df, "sel_cols":sel_cols, "info_cols":info_cols}
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
                tsv = os.path.join(base_dir, dfname + ".tsv")
                if not Path(tsv) or char == "L":
                    df = load_results(dfname)
                    sel_cols = list(df.columns) 
                else:
                    df = pd.read_table(tsv)
                    sel_cols = load_obj("sel_cols", dfname, list(df.columns)) 
                save_obj(dfname, "dfname", "")
                info_cols = load_obj("info_cols", dfname, []) 
                main_df = df
        elif char in list("0123456789"):
            canceled, col = list_values(sel_cols, si=int(char))
            if not canceled:
                sel_cols = order(sel_cols, [col],int(char))
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
        elif char == "j":
            canceled, col = list_values(info_cols)
            if not canceled:
                pos = myinput("pos:","")
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
                sel_cols = order(sel_cols, [sort])
        elif char in ["c","C"]: 
            counts = {}
            for col in df:
               counts[col] = df[col].nunique()
            df = pd.DataFrame(data=[counts], columns = df.columns)
        elif char in ["u","U"]: 
            sel_cols = back["sel_cols"]
            canceled, col = list_values(sel_cols)
            back_df = back["df"]
            if not canceled:
                df = back_df[col].value_counts(ascending=False).reset_index()
            sel_cols = list(df.columns)
            col_widths["index"]=50
            info_cols = []
        elif char in ["g","G"]:
            canceled, col = list_values(sel_cols)
            if not canceled:
               g_cols = [col, "a_bert_score", "a_rouge_score"]
               df = df[g_cols]
               df = df.groupby(col).mean()
               df = df.reset_index()
               sel_cols = order(sel_cols, g_cols)

        elif char in ["d","D"]:
            canceled, col, val = list_df_values(main_df)
            if not canceled:
                main_df = main_df.drop(main_df[main_df[col] == val].index)
                info_cols = []
                if col in df:
                    df = df.drop(df[df[col] == val].index)
                if char == "D":
                    char = "S"
        elif char in ["m","M"]:
            cond = ""
            canceled = False
            sels = []
            info_cols = []
            sel_cols = []
            while not canceled:
                canceled, col, val = list_df_values(main_df, col="model", get_val=True,sels=sels)
                cond += f"| (df['{col}'] == '{val}') "
                info_cols.append("a_input_text_"+val)
                info_cols.append("a_prefix_"+val)
                sel_cols.append("a_pred_text1_"+val)
                sels.append(val)
            cond = cond.strip("|")
            df = main_df[eval(cond)]
            if df.duplicated(['qid','model']).any():
                show_err("There is duplicated rows for qid and model")
                char = "r"
            else:
                df = df.set_index(['qid','model'])[['a_pred_text1', 'a_input_text','a_prefix']].unstack()
                df.columns = list(map("_".join, df.columns))
                for s in sel_cols:
                    col_widths[s] = 35
                store_back = True
        elif char in ["f","F"]:
            canceled, col, val = list_df_values(main_df)
            if not canceled:
                df = main_df[main_df[col] == val]
                df = df.reset_index()
                if char == "F":
                    sel_cols = order(sel_cols, [col])
                sel_row = 0
        elif char == "w":
            canceled, col, val = list_df_values(df, get_val=False)
            if not canceled:
                cmd, _ = minput(cmd_win, 0, 1, ":width=", all_chars=True)
                if cmd.isnumeric():
                    col_widths[col] = int(cmd)
                    save_obj(col_widths, "widths", dfname)
        elif not char in ["q", "S","r"]:
            mbeep()
        if char == "S":
            cmd, _ = minput(cmd_win, 0, 1, "File Name=", default=dfname, all_chars=True)
            if cmd != "<ESC>":
                dfname = cmd
                df.to_csv(os.path.join(base_dir, cmd+".tsv"), sep="\t", index=False)
                save_obj(dfname, "dfname", dfname)
        elif char == "r":
            df = main_df
            sel_cols = list(df.columns)
            info_cols = []
        elif char == "b" and back:
            df = back["df"] 
            sel_cols = back["sel_cols"] 
            info_cols = back["info_cols"]

def myinput(prompt=":", default=""):
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
    key = "_".join(vals[:4])
    if si == 0:
        if key in si_hash:
            si = si_hash[key]
    vals.insert(0,"DONE!")
    opts = {"items":{"sels":sels, "range":vals}}
    is_cancled = True
    si,canceled, _ = open_submenu(tag_win, opts, "items", si, "Select a value", std)
    val = ""
    if not canceled and si > 0: 
        val = vals[si]
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
base_dir = "/home/ahmad/results"
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
    fname = load_obj("dfname","","")
    if fname:
        dfname = fname
        df = pd.read_table(os.path.join(base_dir, dfname + ".tsv"))
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    else:
        df = load_results(dfname)
    show_df(df)

@click.command()
@click.option(
    "--fname",
    "-f",
    default="full_results",
    type=str,
    help=""
)
def main(fname):
    global dfname
    dfname = fname
    set_app("showdf")
    wrapper(start)

if __name__ == "__main__":
    main()