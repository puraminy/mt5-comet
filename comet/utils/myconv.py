import csv
from pathlib import Path
import click
import json
import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import comet.utils.myutils as mu
import comet.utils.myfiles as mf


@click.command()
@click.argument("fname", type=str)
@click.option(
    "--path", envvar="PWD", type=click.Path(),
)
@click.option(
    "--out_format", default="", type=str, help="Output file format (csv,tsv)"
)
@click.option(
    "--prefix", default="", type=str, help="The prefix for output file name"
)
@click.option(
    "--postfix", default="", type=str, help="The postfix for output file name"
)
@click.option("--out_folder", default="", type=str, help="")
@click.option(
    "--in_place",
    "-ip",
    is_flag=True,
    help="If set the output file name is the same fname",
)
@click.option(
    "--filter_none",
    default="",
    type=str,
    help="A column that you want to filter none values",
)
@click.option(
    "--replace",
    default="",
    type=str,
    help="--replace=col-eq | neq-oldvalue-newvalue",
)
@click.option("--split", default=0, type=float, help="")
@click.option(
    "--evenly_per",
    default="",
    type=str,
    help="Sample with equal number for each specified group, ex: --evenly_per=age",
)
@click.option(
    "--prop_to",
    default="",
    type=str,
    help="Sample proportioned to the specified column, ex: --prop_to=age",
)
@click.option("--sample", default=-1, type=int, help="")
@click.option(
    "--addcol",
    default="",
    type=str,
    help="A fixed value column like columnname-fixedvalue or a column from another file like anotherfile@columnname or combination of other columns like columnname-col1+col2",
)
@click.option(
    "--rename",
    default="",
    type=str,
    help="Rename columns example 'oldcol:addcolname, oldcol2:addcol2'",
)
@click.option(
    "--col2file",
    "-f",
    is_flag=True,
    help="If set, it writes --addcol to a file with the name of addcolname",
)
@click.option("--new", "-n", is_flag=True, help="If set, a new file is created")
@click.option(
    "--no_header",
    "-nh",
    is_flag=True,
    help="If set no header is added to the output",
)
@click.option(
    "--groups",
    default="",
    type=str,
    help="Reports the number in each category grouped by --groups",
)
@click.option(
    "--append",
    default="",
    type=str,
    help="the path to another dataframe to append below current dataframe",
)
@click.option(
    "--duplicates",
    default="",
    type=str,
    help="format command:col1-col2...(all means all columsn)  possible values for cammand: remove to remove duplicates, report: to report them",
)
@click.option(
    "--extract_cols",
    default="",
    type=str,
    help="Extract specified columns as new file, format: filename:col1-col2-...",
)
@click.option(
    "--extract_rows",
    default="",
    type=str,
    help="extract rows based on the given condition in format col-operation-value",
)
@click.option("--remove_index", "-ri", is_flag=True, help="")
@click.option(
    "--remove_col",
    default="Dash separated columns to remove",
    type=str,
    help="",
)
@click.option("--info", "-", is_flag=True, help="")
@click.option("--add_index", "-ai", is_flag=True, help="")
@click.option(
    "--index_name", default="", type=str, help="The name for index name"
)
@click.option("--get_index", "-gi", is_flag=True, help="")
@click.option(
    "--joiner",
    default="--",
    type=str,
    help="A string to join two columns in adding new column.",
)
@click.option("--groupby", default="", type=str, help="")
@click.option(
    "--move_col",
    default="",
    type=str,
    help="Move position of a column --move_col=colname:pos",
)
@click.option(
    "--order_cols",
    default="",
    type=str,
    help="Full or partial list of comma separated columns to order columns according",
)
@click.option(
    "--search",
    default="",
    type=str,
    help=""
)
def convert(
    fname,
    path,
    out_format,
    prefix,
    postfix,
    out_folder,
    in_place,
    filter_none,
    replace,
    split,
    evenly_per,
    prop_to,
    sample,
    addcol,
    rename,
    col2file,
    new,
    no_header,
    groups,
    append,
    duplicates,
    extract_cols,
    extract_rows,
    remove_index,
    remove_col,
    info,
    add_index,
    get_index,
    index_name,
    joiner,
    groupby,
    move_col,
    order_cols,
    search,
):
    if not "/" in fname:
        fname = path + "/" + fname
    print("addcol:", addcol)
    if col2file and not addcol:
        print("Please enter your desired column by --addcol option")
        return

    suffix = Path(fname).suffix
    if not out_format:
        ext = ".tsv" if fname.endswith("csv") else ".csv"
    else:
        ext = "." + out_format

    sep = "," if suffix == ".csv" else "\t"
    if not postfix:
        out = path + "/" + Path(fname).stem + "_postfix" + suffix
    elif in_place:
        out = fname
    else:
        out = path + "/" + postfix + suffix
    print("fname:", fname)

    def read_df(fname):
        if fname.endswith("csv"):
            df = pd.read_csv(
                fname,
                keep_default_na=False,
                index_col=0,
                encoding="utf8",
                quoting=csv.QUOTE_ALL,
                error_bad_lines=False,
            )
        else:
            df = pd.read_csv(
                fname,
                sep="\t",
                keep_default_na=False,
                encoding="utf8",
                quoting=csv.QUOTE_ALL,
                error_bad_lines=False,
            )
        return df

    if new:
        df = pd.DataFrame()
        out = fname
    else:
        df = read_df(fname)

    if out_folder:
        path = os.path.join(path, out_folder)
        if Path(path).is_file():
            print(f"A file with out_folder name '{out_folder}' exits!")
            return
        Path(path).mkdir(exist_ok=True, parents=True)
    if get_index:
        il = df.iloc[:, 0]  # list(df.index.values)
        out = path + "/" + Path(fname).stem + "_index"
        print(out)
        with open(out, "w") as f:
            for item in il:
                print(item, file=f)
        return
    if add_index:
        df = df.reset_index()
        df.columns.name = "index"
        df.to_csv(fname, sep=sep)
        return
    if index_name:
        df.columns.name = index_name
        df.to_csv(fname, sep=sep)
        return
    if remove_index:
        df = pd.read_csv(fname, index_col=[0], sep=sep)
        df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
        df.to_csv(fname, index=False, sep=sep)
        return
    if order_cols:
        cols = df.columns.tolist()
        order = [x.strip() for x in order_cols.split(",")]
        for o in order:
            cols.remove(o)
        cols = order + cols
        df = df[cols]
        print("columns:", df.columns)
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")
        return
        df.to_csv(fname, index=False, sep=sep)
        return
    if move_col:
        new_pos = -1
        if ":" in move_col:
            move_col, new_pos = move_col.split(":")
            new_pos = int(new_pos)
        if new_pos >= 0:
            df.insert(new_pos, move_col, df.pop(move_col))
        else:
            df[move_col] = df.pop(move_col)
        print("columns:", df.columns)
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")
        return
        df.to_csv(fname, index=False, sep=sep)
        return
    if info:
        print("columns:", df.columns)
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")
        print("number of rows:", len(df))
        return
    if groupby:
        print(len(df))
        df = (
            df.sort_values("pred1_score", ascending=False)
            .drop_duplicates(["prefix", "input_text"])
            .rename(columns={"target_text": "top"})
            .merge(
                df.groupby(["prefix", "input_text"], as_index=False)[
                    "target_text"
                ].agg("<br />".join)
            )
        )

        out = path + "/grouped_" + Path(fname).stem + suffix
        print(out)
        print(len(df))
        df.to_csv(out, index=False, sep=sep)
        return
    if search:
        pass
    if append:
        if not "/" in append:
            append = path + "/" + append
        df2 = read_df(append)
        df = pd.concat([df, df2], ignore_index=True)
        out = (
            path
            + "/"
            + Path(fname).stem
            + "_concat_"
            + Path(append).stem
            + suffix
        )
        print(out)
        df.to_csv(out, index=False, sep=sep)
        return
    if False:  # remove_col:
        df = df.drop(remove_col.split("-"), axis=1)
        df.to_csv(fname, index=False, sep=sep)
        return
    if duplicates:
        print("size:", len(df))
        dups = duplicates.split(":")
        if dups[0] == "remove" and dups[1] == "all":
            df = df.drop_duplicates()
            print("len after removign dups:", len(df))
            out = path + "/" + Path(fname).stem + "_no_dups" + suffix
        elif dups[0] == "report":
            if dups[1] == "all":
                df = df[df.duplicated(keep=False)]
            else:
                df = df[df.duplicated(subset=dups[1].split("-"), keep=False)]
            print("duplicates:", len(df))
            out = path + "/" + Path(fname).stem + "_dups" + suffix
        else:
            df = df.drop_duplicates(subset=dups[1].split("-"))
            print("len after removign dups:", len(df))
            out = path + "/" + Path(fname).stem + "_no_dups" + suffix

        print("length:", len(df))
        if in_place:
            print("Overwrite:", fname)
            df.to_csv(
                fname,
                sep=sep,
                index=False,
                encoding="utf-8",
                header=not no_header,
            )
        else:
            print(out)
            df.to_csv(
                out,
                sep=sep,
                index=False,
                encoding="utf-8",
                header=not no_header,
            )
        return

    if extract_cols:
        extract = extract_cols.split(":")
        new_df = df[extract[1].split("-")].copy()
        out = path + "/" + extract[0] + suffix
        print(out)
        new_df.to_csv(
            out, sep=sep, index=False, encoding="utf-8", header=not no_header
        )
        return
    if filter_none:
        l1 = len(df)
        print("Before filter:", l1)
        df = df.loc[df[filter_none].str.strip() != "none"]
        l2 = len(df)
        print("After filter:", l2)
        print(l1 - l2, " items were filtered")
        if sample < 0:
            out = path + "/" + Path(fname).stem + "_nn" + suffix
            df.to_csv(
                out,
                sep=sep,
                index=False,
                encoding="utf-8",
                header=not no_header,
            )
            return
    if extract_rows:
        col, opr, val = extract_rows.split("-")
        if opr == "eq":
            df = df.loc[df[col] == val, :]
        else:
            df = df.loc[df[col] != val, :]
        out = path + "/" + Path(fname).stem + "_" + col + "_" + val + suffix
        print(out)
        df.to_csv(
            out, sep=sep, index=False, encoding="utf-8", header=not no_header
        )
        return
    if replace:
        col, opr, oldval, newval = replace.split("-")
        if opr == "eq":
            df.loc[df[col] == oldval, col] = newval
            vlen = len(df[df[col] == newval])
        else:
            df.loc[df[col] != oldval, col] = newval
            vlen = len(df[df[col] == oldval])
        sample_df = df.groupby(col).sample(n=vlen, random_state=1)
        out = (
            path + "/" + Path(fname).stem + "_" + oldval + "_" + newval + suffix
        )
        print(out)
        sample_df.to_csv(
            out, sep=sep, index=False, encoding="utf-8", header=not no_header
        )
        return
    if evenly_per:
        print(f"Sampling {sample} evenly per {evenly_per} ...")
        assert sample > 0, "The number of sample (--sample) must be given"
        sample_df = df.groupby(evenly_per).sample(n=sample, random_state=1)
        tn = "k".join(str(sample).rsplit("000", 1))
        tn = tn.replace("kk", "m")
        if not prefix:
            tn += "_per_" + evenly_per
            out = path + "/" + Path(fname).stem + "_" + tn + suffix
        else:
            out = (
                path
                + "/"
                + prefix
                + "_"
                + tn
                + ("_" if postfix else "")
                + postfix
                + suffix
            )
        print("out:", out)
        sample_df = shuffle(sample_df)
        sample_df.to_csv(
            out, sep=sep, index=False, encoding="utf-8", header=not no_header
        )

        gb = sample_df.groupby(evenly_per)
        dfs = [(key, gb.get_group(key)) for key in gb.groups]
        for gkey, gdf in dfs:
            out = (
                path
                + "/"
                + prefix
                + "_"
                + gkey
                + "_"
                + tn
                + ("_" if postfix else "")
                + postfix
                + suffix
            )
            gdf.to_csv(
                out,
                sep=sep,
                index=False,
                encoding="utf-8",
                header=not no_header,
            )
        return
    if prop_to:
        probs = df[prop_to].map(df[prop_to].value_counts())
        sample_df = df.sample(n=sample, weights=probs)
        tn = "k".join(str(sample).rsplit("000", 1))
        tn = (
            tn.replace("kk", "m")
            + "_propto_"
            + prop_to
            + ("_nn" if filter_none else "")
            + suffix
        )
        out = path + "/" + Path(fname).stem + "_" + tn
        print("out:", out)
        sample_df.to_csv(
            out, sep=sep, index=False, encoding="utf-8", header=not no_header
        )
        return
    if groups:
        counts = df.value_counts(groups.split(","))
        percents = counts / len(df)
        gs = counts.keys()
        for g, c, p in zip(gs, counts, percents):
            print(g, ":", c, f"{p:.2f}%")

        return
    if split > 0:
        train_fname = path + "/" + Path(fname).stem + "_train" + suffix
        test_fname = path + "/" + Path(fname).stem + "_test" + suffix
        train, test = train_test_split(df, test_size=split)

        train.columns = df.columns
        test.columns = df.columns
        train.to_csv(
            train_fname,
            sep=sep,
            index=False,
            encoding="utf-8",
            header=not no_header,
        )
        test.to_csv(
            test_fname,
            sep=sep,
            index=False,
            encoding="utf-8",
            header=not no_header,
        )
        print(train_fname)
        print(test_fname)
        return

    if rename:
        res = mu.arg2dict(rename)
        df = df.rename(columns=res)
        if not addcol:
            df.to_csv(
                fname,
                sep=sep,
                index=False,
                encoding="utf-8",
                header=not no_header,
            )
            print(res, " were renamed")
            return

    if addcol:
        if not "@" in addcol:
            name_value = addcol.split("-")
            addcol_name = name_value[0]
            newval = name_value[1]
            if newval in df.columns:
                newval = df[newval]
            elif "+" in newval:
                cols = newval.split("+")
                newval = df[cols].agg(joiner.join, axis=1)
        else:
            name_value = addcol.split("@")
            of_name = name_value[0]
            addcol_name = name_value[1]
            suffix = Path(path + "/" + of_name).suffix
            if suffix == ".jsonl":
                data = mf.read_jsonl(path + "/" + of_name)
                newval = []
                for row in data:
                    newval.append(row[addcol_name])
            elif suffix in [".csv", ".tsv"]:
                odf = read_df(of_name)
                assert len(df) == len(odf), "Lengths must be equal."
                newval = odf[addcol_name].tolist()
        if not col2file:
            df[addcol_name] = newval
            df.to_csv(
                fname,
                sep=sep,
                index=False,
                encoding="utf-8",
                header=not no_header,
            )
            print(name_value, " was added")
        else:
            with open(path + "/" + addcol_name, "w") as f:
                for item in newval:
                    f.write("%s\n" % item)
            print(name_value, " was written")
        return

    if sample > 0:
        df = df.sample(n=sample)
        tn = "k".join(str(sample).rsplit("000", 1))
        tn = tn.replace("kk", "m") + suffix
        out = path + "/" + Path(fname).stem + "_" + tn
        print("out:", out)
        df.to_csv(
            out, sep=sep, index=False, encoding="utf-8", header=not no_header
        )
        return

    print("out:", out)
    if ext == ".tsv":
        with open(out, "w") as tsvout:
            tsvout = csv.writer(tsvout, delimiter="\t")
            for index, row in tqdm(df.iterrows(), total=len(df)):
                tsvout.writerow(row)
    elif ext == ".csv":
        with open(out, "w") as fou:
            cw = csv.writer(fou, escapechar="\\")
            for index, row in tqdm(df.iterrows(), total=len(df)):
                cw.writerow(row)
    elif ext == ".json":
        dlist = []
        for index, row in tqdm(df.iterrows(), total=len(df)):
            d = {}
            item = {}
            en = str(row["en"]).replace('"', "")
            fa = str(row["fa"]).replace('"', "")
            if len(en.strip()) < 15 or len(fa.strip()) < 15:
                continue
            item["en"] = en
            item["fa"] = fa
            d["translation"] = item
            dlist.append(d)
        dd = {"data": dlist}
        with open(out, "w", encoding="utf8") as jsonfile:
            json.dump(dd, jsonfile, ensure_ascii=False)
    else:
        print("Unsuported output format")


if __name__ == "__main__":
    convert()
