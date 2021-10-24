import pandas as pd
from pathlib import Path
from comet.evaluation.myeval import *


def make_link(label, fid, dpath, caption):
    tag = fid + "@" + label.lower().replace(" ", "_")
    return f'<a view="{fid}" cmd="replace_df" href="/" class="df-link" tag="{tag}" path="{dpath}">Browse</a>'


def make_cmd_link(cmd, caption="Recompute"):
    cmd = "calc_" + cmd.lower().replace(" ", "_")
    return f'<button class="loading btn btn-primary" label="{caption}" cmd="{cmd}">{caption}</button>'


nones = ["none", "هیچ یک"]
total_items = 1
unique_items = 1
row1 = {}
row2 = {}
row3 = {}
row4 = {}


def fill_rows(label, df, fid, dpath):
    global row1, row2, row3, row4

    df.to_csv(dpath, sep="\t")
    c = len(df)
    row1[label] = c
    row2[label] = "{:.2f}".format(c / total_items)
    row3[label] = "{:.2f}".format(c / unique_items)
    row4[label] = make_link(label, fid, dpath, c)


def match(df, col1, col2):
    return df[df.apply(lambda x: x[col1] in x[col2], axis=1)]


def make_report(path, fid, df, metrics, match_list, cur_report=""):
    global total_items, unique_items, row1, row2, row3, row4

    row1 = {}
    row2 = {}
    row3 = {}
    row4 = {}
    pred_text1 = match_list[0]
    target_text = match_list[1]

    for field in match_list:
        df[field] = df[field].astype(str)
    if Path(cur_report).is_file():
        rdf = pd.read_table(cur_report, index_col=0)
        row1 = rdf.iloc[:, 0].to_dict()
        row2 = rdf.iloc[:, 1].to_dict()
        row3 = rdf.iloc[:, 2].to_dict()
        row4 = rdf.iloc[:, 3].to_dict()
    else:
        dpath = f"{path}/all_{fid}.tsv"
        df.to_csv(dpath, sep="\t")


        total_items = len(df)
        unique_items = df["input_text"].nunique()

        fill_rows("Total rows", df, fid, dpath)
        tdf = df.groupby("input_text").first()
        dpath = f"{path}/unique_{fid}.tsv"
        fill_rows("Unique events", tdf, fid, dpath)

        tdf = df[
            df.duplicated(subset=["input_text", "target_text"], keep=False)
        ]
        dpath = f"{path}/dups_en_{fid}.tsv"
        fill_rows("Duplicates", tdf, fid, dpath)

        tdf = match(df, pred_text1, target_text)
        dpath = f"{path}/match_en_{fid}.tsv"
        fill_rows("Matches", tdf, fid, dpath)

        tdf = tdf[tdf.target_text != "none"]
        dpath = f"{path}/match_en_{fid}.tsv"
        fill_rows("Not None Matches", tdf, fid, dpath)

        tdf = df[df[pred_text1].str.contains("none", na=False)]
        fill_rows(
            "None Predictions",
            tdf,
            fid,
            dpath=f"{path}/none_preds_en_{fid}.tsv",
        )

        tdf = df[df[target_text].str.contains("none", na=False)]
        fill_rows(
            "None Targets", tdf, fid, dpath=f"{path}/none_targets_en_{fid}.tsv"
        )

    if metrics:
        refs = {}
        hyps = {}
        for idx, row in df.iterrows():
            refs[row["input_text"]] = row[target_text].split("<br />")
            hyps[row["input_text"]] = row[pred_text1].split("<br />")

        QGEval = QGEvalCap("", refs, hyps, metrics)
        scores, _ = QGEval.evaluate(model_type="microsoft/deberta-xlarge-mnli")

        for k, v in scores.items():
            k = k.title()
            row1[k] = len(refs)
            row3[k] = v

    match_df = pd.DataFrame(columns=row1.keys())
    match_df = match_df.append(row1, ignore_index=True)
    match_df = match_df.append(row2, ignore_index=True)
    match_df = match_df.append(row3, ignore_index=True)
    match_df = match_df.append(row4, ignore_index=True)

    cols = ["Items", "Per Total", "Per Unique Events", "Browse"]
    match_df.index = pd.Index(cols, name="Report")
    retdf = match_df.T
    retdf.loc[:, "Items"] = retdf.loc[:, "Items"].astype(int)

    return retdf


def compare(path, fid, fid2, base_df, new_df, cur_report):
    global total_items, unique_items, row1, row2, row3, row4

    row1 = {}
    row2 = {}
    row3 = {}
    row4 = {}
    if Path(cur_report).is_file():
        rdf = pd.read_table(cur_report, index_col=0)
        row1 = rdf.iloc[:, 0].to_dict()
        row2 = rdf.iloc[:, 1].to_dict()
        row3 = rdf.iloc[:, 2].to_dict()
        row4 = rdf.iloc[:, 3].to_dict()
    else:
        left = base_df
        right = new_df
        if "pred_text1_y" in base_df:
            left = base_df.drop(columns=["pred_text1_y", "pred1_score_y"])
        if "pred_text1" in new_df:
            right = new_df.rename(
                columns={
                    "pred_text1": "pred_text1_y",
                    "pred1_score": "pred1_score_y",
                }
            )

        df = pd.merge(left, right)
        df["target_text"] = df["target_text"].astype(str)
        df["pred_text1"] = df["pred_text1"].astype(str)
        df["pred_text1_y"] = df["pred_text1_y"].astype(str)

        total_items = 100 #len(df)
        unique_items = 100 #df["input_text"].nunique()

        dpath = f"{path}/compare_{fid}.tsv"
        fill_rows(fid + " (baseline)", base_df, fid, dpath)
        dpath = f"{path}/compare_{fid}.tsv"
        fill_rows(fid2 + " (new)", new_df, fid, dpath)

        tdf = match(df, "pred_text1", "target_text")
        dpath = f"{path}/compare_match_{fid}.tsv"
        fill_rows(fid + " Matches", tdf, fid, dpath)

        tdf = tdf[df.target_text != "none"]
        dpath = f"{path}/compare_nn_match_{fid}.tsv"
        fill_rows(fid + " Not None Matches", tdf, fid, dpath)

        tdf = match(df, "pred_text1_y", "target_text")
        dpath = f"{path}/compare_match_newmethod_{fid2}.tsv"
        fill_rows(fid2 + " Matches", tdf, fid, dpath)

        tdf = tdf[df.target_text != "none"]
        dpath = f"{path}/compare_nn_match_newmethod_{fid2}.tsv"
        fill_rows(fid2 + " Not None Matches", tdf, fid, dpath)

        tdf = df[
            (df.apply(lambda x: x.pred_text1_y in x.target_text, axis=1))
            & (df.apply(lambda x: x.pred_text1 not in x.target_text, axis=1))
            & ~df["target_text"].isin(nones)
        ]
        dpath = f"{path}/b2_{fid2}.tsv"
        fill_rows(fid2 + " is Better", tdf, fid, dpath)

        tdf = df[
            (df.apply(lambda x: x.pred_text1_y not in x.target_text, axis=1))
            & (df.apply(lambda x: x.pred_text1 in x.target_text, axis=1))
            & ~df["target_text"].isin(nones)
        ]
        dpath = f"{path}/b1_{fid}.tsv"
        fill_rows(fid + " is Better", tdf, fid, dpath)

        tdf = df[
            (df["pred_text1"] == df["pred_text1_y"])
            & (df.apply(lambda x: x.pred_text1 in x.target_text, axis=1))
        ]
        fill_rows(
            "All Common Matches",
            tdf,
            fid,
            dpath=f"{path}/all_commom_match_{fid}.tsv",
        )

        tdf = df[
            (df["pred_text1"] == df["pred_text1_y"])
            & (df.apply(lambda x: x.pred_text1 in x.target_text, axis=1))
            & ~df["pred_text1"].isin(nones)
        ]
        fill_rows(
            "Not None Common Matches",
            tdf,
            fid,
            dpath=f"{path}/nn_commom_match_{fid}.tsv",
        )

        tdf = df[
            (df["pred_text1"] == df["pred_text1_y"])
            & ~df["pred_text1"].isin(nones)
        ]
        tdf = tdf.drop_duplicates(subset=["input_text", "pred_text1"])
        fill_rows(
            "Not None Common Predictions",
            tdf,
            fid,
            dpath=f"{path}/nn_commom_{fid}.tsv",
        )
        tdf = df[
            (df["pred_text1"] == df["pred_text1_y"])
            & df["pred_text1"].isin(nones)
        ]
        fill_rows(
            "None Common Predictions",
            tdf,
            fid,
            dpath=f"{path}/none_commom_{fid}.tsv",
        )

    match_df = pd.DataFrame(columns=row1.keys())
    match_df = match_df.append(row1, ignore_index=True)
    match_df = match_df.append(row2, ignore_index=True)
    match_df = match_df.append(row3, ignore_index=True)
    match_df = match_df.append(row4, ignore_index=True)
    retdf = match_df.T
    retdf.columns = ["Items", "Per Total", "Per Unique Events", ""]
    return retdf
