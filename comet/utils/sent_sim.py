from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
import pandas as pd
from tqdm import tqdm
import glob
import torch

def calc(df, before, after, col1, col2, score_col, cpu):
  if after > 0:
    df = df.truncate(before=before, after=after)
  else:
    df = df.truncate(before=before)
  
  col1_val = df[col1].astype(str).tolist()
  col2_val = df[col2].astype(str).tolist()
  
  device = torch.device("cuda")
  if cpu:
      device = torch.device("cpu")
  col1_val_emb = model.encode(col1_val, device=device, convert_to_tensor=True)
  col2_val_emb = model.encode(col2_val,  device=device, convert_to_tensor=True)
  
  #Compute cosine-similarits
  cosine_scores = util.pytorch_cos_sim(col1_val_emb, col2_val_emb)
  
  #Output the pairs with their score
  scores = []
  for i in tqdm(range(len(df)), total = len(df)):
      scores.append("{:.4f}".format(cosine_scores[i][i]))
  
  df[score_col] = scores
  return df


from pathlib import Path
import click
@click.command()
@click.argument("pred_pat", type=str)
@click.option(
    "--fname",
    default="src_df.tsv",
    type=str,
    help=""
)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--step",
    default=1000,
    type=int,
    help=""
)
@click.option(
    "--col1",
    default="pred_text1",
    type=str,
    help=""
)
@click.option(
    "--col2",
    default="target_text",
    type=str,
    help=""
)
@click.option(
    "--score_col",
    default="pred1_score",
    type=str,
    help=""
)
@click.option(
    "--cpu",
    "-",
    is_flag=True,
    help=""
)
@click.option(
    "--no_concat",
    "-",
    is_flag=True,
    help=""
)
@click.option(
    "--addinp",
    "-a",
    is_flag=True,
    help=""
)
def main(fname, pred_pat, path, step, col1, col2, score_col, cpu, no_concat, addinp):
    concat = not no_concat
    if not Path(fname).exists():
        if "val" in pred_pat:
            fname = "/drive3/pouramini/data/atomic/en_fa/xIntent_en_fa_validation_no_dups.tsv"
        else:
            fname = "/drive3/pouramini/data/atomic/en_fa/xIntent_en_fa_train_no_dups.tsv"
    if fname.endswith("csv"):
        srcdf = pd.read_csv(fname)
    else:
        srcdf = pd.read_table(fname)
        
    if pred_pat != "none":
        inps = glob.glob(f"{path}/*{pred_pat}*predictions")
        if len(inps) == 0:
            print(f"A file with this pattern '{pred_pat}*predictions' wasn't found")
            return
        pred_file = inps[0]
        inps = glob.glob(f"{path}/*inputs")
        input_file = inps[0]
        inpcol=[]
        f = open(input_file, "r")
        ncc = 0

        def persianReplace(text):
            text = text.strip().replace("PersonX's", "خود").replace("PersonY", "رضا")
            text = text.strip().replace("PersonX", "علی").replace("PersonZ", "حمید")
            return text

        fa = ""
        if "fa" in Path(path).stem:
            fa = "fa"
        for x in f:
            text = x
            if "fa" in Path(path).stem:
                text = text[2:-1]
                text = text.encode("raw_unicode_escape")
                text = text.decode("unicode_escape")
                text = text.encode("raw_unicode_escape")
                text = text.decode()
                text = persianReplace(text)
            text = text.strip()
            inpcol.append(text)

        print("Prediction file:", pred_file)
        with open(pred_file) as f:
            preds = f.read().splitlines()
        
        print("Prediction Lengths:", len(preds))
        if not "/" in fname:
            fname = path + "/" + fname
        if not Path(fname).is_file():
            print(f"A file with the name {fname} wasn't found")
            return
        if len(preds) == len(srcdf) + 1:
            preds = preds[1:]
            inpcol = inpcol[1:]
        print("len preds:", len(preds))
        print("len srcdf:", len(srcdf))
        print("len inpcol:", len(inpcol))
        assert len(preds) == len(srcdf), "The length of source and prediction files must be equal"
        srcdf[f"input_text_prompted{fa}"] = inpcol
        srcdf[col1] = preds
    before = 0
    after = step
    out = path + "/scored_" + col1 + "_" + fname.replace(".csv", ".tsv")
    if after < 0:
        print(out)
        calc(srcdf, before, -1, col1, col2, score_col, cpu)
    else:
        #Path(out).mkdir(exist_ok=True, parents=True)
        df_old = None
        while True:
          print(before, "-", after)
          df = calc(srcdf, before, after, col1, col2, score_col, cpu)
          #out = path + f"/scored_{before}_{after}_{col1}_" + Path(fname).stem  + ".tsv" 
          #print(out)
          #print(len(df))
          #df.to_csv(out, sep="\t", index=False, header=False)
          if df_old is not None:
              df = pd.concat([df_old, df], ignore_index=True)
          df_old = df
          before = after + 1
          after += step
          if after >= len(srcdf):
              after = -1
          if before == 0:
              break
    
    df[col2] = df[col2].astype(str)
    if concat:
        df = df.sort_values(score_col, ascending=False).\
          drop_duplicates(['prefix','input_text']).\
            rename(columns={col2:'top'}).\
              merge(df.groupby(['prefix','input_text'],as_index=False)[col2].agg('<br />'.join))

    out = path + "/scored_" + col1 + "_" + Path(fname).stem  + ".tsv" 
    print(out)
    print(len(df))
    df.to_csv(out, sep="\t", index=False)

if __name__ == "__main__":
    main()
