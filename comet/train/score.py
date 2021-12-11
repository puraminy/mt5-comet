from sentence_transformers import SentenceTransformer, util
#model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
from comet.train.bart_score import BARTScorer
from comet.train.common import *


import pandas as pd
from tqdm import tqdm
import glob
import torch
def bart_score(model, df, before, after, col1, col2, score_col, cpu):
  if after > 0:
    df = df.truncate(before=before, after=after)
  else:
    df = df.truncate(before=before)


  scores = []
  pbar = tqdm(total = len(df))
  for i, row in df.iterrows(): 
      s1 = row[col1]
      s2 = row[col2]
      rel = row.prefix
      langs = row.langs
      from_, to_ = langs.split("2")
      s2 = relation_natural_mappings[rel][to_] + " " + s2
      if i < 5:
          print(i)
          print(s1)
          print(s2)
      pbar.update()
      score = model.score([s1], [s2], batch_size=4) # generation scores from the first list of texts to the second list of texts.
      scores.append(":.5f".format(score[0]))
  
  df[score_col] = scores
  return df

def bert_score(model, df, before, after, col1, col2, score_col, cpu):
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
@click.argument("fname", type=str)
@click.option(
    "--path",
    "-p",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--model_id",
    "-m",
    default="",
    type=str,
    help=""
)
@click.option(
    "--step",
    default=-1,
    type=int,
    help=""
)
@click.option(
    "--col1",
    "-c1",
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
    "-sc",
    default="bart_score",
    type=str,
    help=""
)
@click.option(
    "--cpu",
    "-cpu",
    is_flag=True,
    help=""
)
@click.option(
    "--concat",
    "-c",
    is_flag=True,
    help=""
)
def main(fname, model_id, path, step, col1, col2, score_col, cpu, concat):
    pret = "/content/drive/MyDrive/pret"
    if "bert_score" in score_col:
        if not model_id: model_id = '/paraphrase-multilingual-MiniLM-L12-v2'
        model = SentenceTransformer(os.path.join(pret, model_id))
    else:
        device = "cpu" if cpu else "cuda:0"
        if not model_id: model_id= "enfat5-large"
        model = BARTScorer(device=device, checkpoint=os.path.join(pret, model_id))

    score_col = model_id + "_" + score_col
    if fname.endswith("csv"):
        srcdf = pd.read_csv(fname)
    else:
        srcdf = pd.read_table(fname)
        
    before = 0
    after = step
    if after < 0:
        if "bert_score" in score_col:
            df = bert_score(model, srcdf, before, -1, col1, col2, score_col, cpu)
        else:
            df = bart_score(model, srcdf, before, -1, col1, col2, score_col, cpu)
    else:
        df_old = None
        while True:
          print(before, "-", after)
          if "bert_score" in score_col:
             df = bert_score(model,srcdf, before, after, col1, col2, score_col, cpu)
          else:
             df = bart_score(model,srcdf, before, after, col1, col2, score_col, cpu)
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

    out1 = resPath + "/"+ score_col + "_" + col1 + "_" + Path(fname).stem  + ".tsv" 
    out2 = logPath + "/"+ score_col + "_" + col1 + "_" + Path(fname).stem  + ".tsv" 
    df.to_csv(out1, sep="\t", index=False)
    df.to_csv(out2, sep="\t", index=False)
    print(len(df))
    out1 = out1.replace("/content/drive/MyDrive/", "")
    print("rcopy nlp:{out1} .)

if __name__ == "__main__":
    main()
