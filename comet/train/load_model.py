from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast, AutoTokenizer 
    )
import click
from termcolor import colored
from comet.train.common import *
from comet.train.eval import *
import pandas as pd
import random

@click.command()
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
def main(path):
    print("Loading model...")
    underlying_model_name = path
    tokenizer = AutoTokenizer.from_pretrained(underlying_model_name)
    model = T5ForConditionalGeneration.from_pretrained(underlying_model_name)

    my_spacials = [x for x in tokenizer.additional_special_tokens if not "<extra_id"  in x]
    print(my_spacials)
    doc = ""
    input_text = ""
    df = pd.read_table("/home/pouramini/results/sel/merged.tsv", low_memory=False)
    df = df[["input_text","date", "method","prefix","pred_text1","model", "wrap","target_text", "rouge_score", "bert_score"]]
    df["target_text"] = df["target_text"].astype(str)
    score_col = "rouge_score"
    group_col = "target_text"
    df = df.sort_values(score_col, ascending=False).\
         drop_duplicates(['date','prefix','input_text']).\
            rename(columns={group_col:'top_target'}).\
              merge(df.groupby(['date','prefix','input_text'],as_index=False)[group_col].agg(' | '.join))
    rand_inp = ""
    docs = []
    ii =1
    while doc != "end":
        print(colored("------------------------------------------------","red"))
        print("%=Random:", rand_inp)
        inp = input(f" :") 
        if not inp:
            r = random.randint(0, len(df) -1)
            inp = "%"
            rand_inp = str(df.iloc[r]["input_text"])
            input_text = rand_inp
        if inp in relation_natural_mappings:
            inp = "#" + inp

        if inp.startswith("@"):
            input_text = inp.replace("@","")
        if "$" in inp:
            inp = inp.split("$")[1]
            doc = docs[int(inp)]
        elif "=" in inp:
            old,new = inp.split("=")
            doc = doc.replace(old,new).replace("<>","")
        else:
            doc = inp.replace("#",input_text).replace("!",doc).replace("@","").replace("%",rand_inp)
        docs.append(doc)
        prefix = "" 
        for key, item in relation_natural_mappings.items():
            if key in doc:
                prefix = key
            doc = doc.replace(key, item["tokens"]) 

        print(f"{ii:<2})", colored(doc, 'green'))
        ii += 1
        # encode input context
        # generate 3 independent sequences using beam search decoding (5 beams)
        # with T5 encoder-decoder model conditioned on short news article.
        #outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
        outputs = gen_resp(model, tokenizer, doc)
        print("Generated:", colored(outputs,'blue'))
        if rand_inp in doc or doc == rand_inp:
            print(colored("------------------------------------------------","yellow"))
            preds = df[(df["input_text"] == rand_inp) & (df["prefix"] == prefix)]
            first = True
            for idx, row in preds.iterrows():
                if first:
                    first = False
                    print(colored(row["target_text"],"blue"))
                print("{:<50}:{:<4} {:<4} {}".format(
                    colored(row["model"]+"|"+row["method"]+"|"+row["wrap"],'yellow'),
                    colored(round(row["rouge_score"],2),"green"),
                    colored(round(row["bert_score"],2),"blue"),
                    row["pred_text1"]))

main()
