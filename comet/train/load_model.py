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
@click.argument("fname", type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--load_model",
    "-l",
    is_flag=True,
    help=""
)
def main(fname, path, load_model):
    if load_model:
        print("Loading model...")
        underlying_model_name = path
        tokenizer = AutoTokenizer.from_pretrained(underlying_model_name)
        model = T5ForConditionalGeneration.from_pretrained(underlying_model_name)
        #model.load_state_dict(torch.load(path + "/sate_dict"))
        model.eval()

        my_spacials = [x for x in tokenizer.additional_special_tokens if not "<extra_id"  in x]
        print(my_spacials)


    pd.options.mode.chained_assignment = None
    doc = ""
    input_text = ""
    df = pd.read_table(f"/home/pouramini/results/sel/{fname}.tsv", low_memory=False)
    df = df[["input_text","date", "fid", "method","prefix","pred_text1","model", "wrap","target_text", "rouge_score", "bert_score"]]
    df["target_text"] = df["target_text"].astype(str)
    score_col = "rouge_score"
    group_col = "target_text"
    df = df.sort_values(score_col, ascending=False).\
         drop_duplicates(['fid','prefix','input_text']).\
            rename(columns={group_col:'top_target'}).\
              merge(df.groupby(['fid','prefix','input_text'],as_index=False)[group_col].agg(' | '.join))
    rand_inp = ""
    docs = []
    ii =1
    preds = None
    filt_preds = None
    prefix = ""
    sort = "rouge_score"
    filt = ""
    gen_token=""
    #wwwwwwwww
    while doc != "q":
        print(colored("? (help) --------------------------------------","red"))
        print("Input:", colored(rand_inp,'cyan'), f" [{prefix}]")
        inp = input(f" :") 
        if not inp:
            r = random.randint(0, len(df) -1)
            inp = "%"
            rand_inp = str(df.iloc[r]["input_text"])
            filt=""
            filt_preds = None
        if inp.isnumeric():
            inp = list(relation_natural_mappings.keys())[int(inp)] 
        if inp.startswith("?") or inp == "help":
            if inp == "help": inp = "?"
            ch = inp.split("?")[1]
            if ch == "":
                print(colored("Enter",'cyan'), ": get random input")
                print(colored("?rels",'cyan'), ": show numbers for relation")
                print(colored("#",'cyan'), ": place holder for the previous input")
            if ch == "rels":
                for i, key in enumerate(list(relation_natural_mappings.keys())):
                    print(colored(i,"cyan").strip(),":", key, end=" ")
                    if i % 4 == 0:
                        print("")
                print("")
            continue
        if inp.startswith("~"):
            gen_token = inp[1:]
            inp = "!"
        if inp in relation_natural_mappings:
            inp = "#" + inp
        if inp.startswith("*"):
            filt = inp.split("*")[1]
        if inp.startswith("^"):
            sort = inp.split("^")[1]
            if sort == "r": sort = "rouge_score"
            if sort == "b": sort = "bert_score"

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
        if load_model:
            outputs = generate(model, tokenizer, [doc], 1, gen_token)
            print("Generated:", colored(outputs,'blue'))
        if filt and preds is not None:
            filt_preds = preds.loc[preds["fid"].str.contains(filt),:]
        elif prefix and rand_inp and rand_inp in doc:
            print(prefix, ":", colored("---------------------------------------","yellow"))
            preds = df.loc[(df["input_text"] == rand_inp) & (df["prefix"] == prefix),:]
            filt_preds = preds.loc[:,:]

        if inp != "q" and filt_preds is not None:
            filt_preds.sort_values(by=sort,ascending=False,inplace=True)
            first = True
            for idx, row in filt_preds.iterrows():
                if first:
                    first = False
                    print(colored(row["target_text"],"blue"))
                print("{:<60}:{:<4} {:<4} {}".format(
                    #colored(row["model"]+"|"+row["method"]+"|"+row["wrap"],'yellow'),
                    colored(row["fid"],'yellow'),
                    colored(round(row["rouge_score"],2),"green"),
                    colored(round(row["bert_score"],2),"blue"),
                    row["pred_text1"]))

main()
