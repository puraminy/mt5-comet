from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast, AutoTokenizer 
    )
import click
from termcolor import colored
from comet.train.common import *
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
    df = pd.read_table("/home/pouramini/mt5-comet/comet/train/atomic/val_all_rels.tsv")
    rand_inp = ""
    docs = []
    ii =1
    while doc != "end":
        print(colored("------------------------------------------------","red"))
        print("%=Random:", rand_inp)
        inp = input(f"[#={input_text}]:") 
        if inp == "":
            r = random.randint(0, len(df) -1)
            rand_inp = df.iloc[r]["input_text"] 

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

        print(ii,")", colored(doc, 'green'))
        ii += 1
        # encode input context
        doc_tokens = tokenizer.tokenize(doc)
        #print("doc_tokens:", doc_tokens)
        doc_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
        input_ids = tokenizer(doc, return_tensors="pt").input_ids
        #print("doc_ids:", doc_ids)
        #print("input_ids:", input_ids)
        # generate 3 independent sequences using beam search decoding (5 beams)
        # with T5 encoder-decoder model conditioned on short news article.
        outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
        print("Generated:", colored(tokenizer.batch_decode(outputs, skip_special_tokens=True),'blue'))
        if rand_inp in doc:
            print("Target:", df.loc[(df["input_text"] == rand_inp) & (df["prefix"] == prefix), "target_text"]) 

main()
