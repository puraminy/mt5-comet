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
  rows = []
  for i, row in df.iterrows(): 
      rel = row.prefix
      data = {}
      langs = row.langs
      from_, to_ = langs.split("2")
      s1 = row.input_text
      s1 = s1.replace("##","").replace("  "," ").strip()
      s2 = row[col1]
      s2 = relation_natural_mappings[rel][to_] + " " + s2
      data["s1"] = s1
      data["s2"] = s2
      data["langs"] = langs
      pbar.update()
      score = model.score([s1], [s2], batch_size=4) # generation scores from the first list of texts to the second list of texts.
      scores.append("{:.5f}".format(score[0]))
      data["score"] = score
      rows.append(data)
      if i < 5:
          print(i)
          print(s1)
          print(s2)
          print(score)
  
  df[score_col] = scores
  new_df = pd.DataFrame(data=rows)
  return df, new_df

def bert_score(model, df, before, after, col1, col2, score_col, cpu):
  if after > 0:
    df = df.truncate(before=before, after=after)
  else:
    df = df.truncate(before=before)
  
  device = torch.device("cuda")
  if cpu:
      device = torch.device("cpu")

  data_rows = []
  for idx, row in tqdm(df.iterrows(), total=len(df)):
        data = {}
        data["s1"] = row[col1]
        data["s2"] = row[col2]
        sents1 = row[col1].split("<br />")
        sents2 = row[col2].split("<br />")

        #Compute embeddings
        embeddings1 = model.encode(sents1, device=device, convert_to_tensor=True)
        embeddings2 = model.encode(sents2, device=device, convert_to_tensor=True)

        #Compute cosine-similarities for each sentence with each other sentence
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        #print(cosine_scores)

        #Find the pairs with the highest cosine similarity scores
        pairs = []
        rows = cosine_scores.shape[0]
        cols = cosine_scores.shape[1]
        for i in range(rows):
            for j in range(cols):
                pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
                #print({'index': [i, j], 'score': cosine_scores[i][j]})

        #Sort scores in decreasing order
        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

        top = pairs[0]

        data["target"] = df.loc[idx, "best_target"] = sents2[top["index"][0]]
        data["prediction"] = df.loc[idx, "best_pred"] = sents2[top["index"][1]]
        data[score_col] = df.loc[idx, score_col] = "{:.4f}".format(top["score"])
        data_rows.append(data)

  new_df = pd.DataFrame(data=data_rows)

  return df,new_df


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
    "--model_name",
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
    default="bert_score",
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
def main(fname, model_name, path, step, col1, col2, score_col, cpu, concat):
    pret = "/content/drive/MyDrive/pret"
    if not colab:
        pret = os.path.join(home, "pret")
    if "bert_score" in score_col:
        if not model_name: model_name = 'paraphrase-MiniLM-L6-v2'
        model_path = os.path.join(pret, model_name)
        if Path(model_path).exists():
            model_name = model_path
        model = SentenceTransformer(os.path.join(pret, model_name))
    else:
        device = "cpu" if cpu else "cuda:0"
        if not model_name: model_name= "t5-large"
        model_path = os.path.join(pret, model_name)
        if Path(model_path).exists():
            model_name = model_path

        model = BARTScorer(device=device, checkpoint=model_name)

    score_col = Path(model_name).stem.replace("/","_") + "_" + col1 + "_" + score_col
    mlog.info("score_col: %s", score_col)
    mlog.info("col1: %s", col1)
    mlog.info("col2: %s", col2)
    if fname.endswith("csv"):
        srcdf = pd.read_csv(fname)
    else:
        srcdf = pd.read_table(fname)

    if score_col in srcdf:
        mlog.info("%s exits, removing it...", score_col)
        del srcdf[score_col]
        
    before = 0
    after = step
    if after < 0:
        if "bert_score" in score_col:
            df,new_df = bert_score(model, srcdf, before, -1, col1, col2, score_col, cpu)
        else:
            df,new_df = bart_score(model, srcdf, before, -1, col1, col2, score_col, cpu)
    else:
        df_old = None
        new_df_old = None
        while True:
          print(before, "-", after)
          if "bert_score" in score_col:
             df,new_df = bert_score(model,srcdf, before, after, col1, col2, score_col, cpu)
          else:
             df,new_df = bart_score(model,srcdf, before, after, col1, col2, score_col, cpu)
          if df_old is not None:
              df = pd.concat([df_old, df], ignore_index=True)
              new_df = pd.concat([new_df_old, new_df], ignore_index=True)
          df_old = df
          new_df_old = new_df
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

    out1 = resPath + "/" + Path(fname).stem  + ".tsv" 
    out2 = logPath + "/" + Path(fname).stem  + ".tsv" 
    out_new1 = resPath + "/new_" + Path(fname).stem  + ".tsv" 
    out_new2 = logPath + "/new_" + Path(fname).stem  + ".tsv" 
    df.to_csv(out1, sep="\t", index=False)
    df.to_csv(out2, sep="\t", index=False)
    new_df.to_csv(out_new1, sep="\t", index=False)
    new_df.to_csv(out_new2, sep="\t", index=False)
    print(len(df))
    out1 = out1.replace("/content/drive/MyDrive/", "")
    print(f"rcopy nlp:{out1} . ")

if __name__ == "__main__":
    main()
