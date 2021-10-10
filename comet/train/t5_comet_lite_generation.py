#%% load libraries

import json
from datasets import load_dataset,load_metric
from transformers import (
    MT5ForConditionalGeneration, MT5TokenizerFast
)
import os,time
import argparse
from tqdm.auto import tqdm
import pandas as pd
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--load_path', default="logs/atomic-mt5/last", type=str,
                    help='path to load saved model')
parser.add_argument('--val_set', default="validation", type=str,
                    help='the set to evaluate on')
args = parser.parse_args()
load_path = args.load_path
val_set = args.val_set
#%% some hyper-parameters
num_generations = 0 #0 means no limit
generation_params = {
    "max_length":80,
    "early_stopping":True
}
#%% load atomic data
#atomic_dataset = load_dataset('atomic')

atomic_dataset = {}
atomic_dataset["train"] = pd.read_table("/drive3/pouramini/data/atomic/en_fa/xIntent_en_train_no_dups.tsv")
atomic_dataset["validation"] = pd.read_table("/drive3/pouramini/data/atomic/en_fa/xIntent_en_fa_validation_no_dups.tsv")

atomic_relation_mappings = {
    "oEffect":"<oEffect>",
    "oReact":"<oReact>",
    "oWant":"<oWant>",
    "xAttr":"<xAttr>",
    "xEffect":"<xEffect>",
    "xIntent":"<xIntent>",
    "xNeed":"<xNeed>",
    "xReact":"<xReact>",
    "xWant":"<xWant>"
}
gen_token = "<gen>"
#%% Aggregate instances of queries and corresponding responses
# (str)split_name -> (dict) query -> (list) response 
print("building query responses")
atomic_query_responses = {}
for split_name,split_data in atomic_dataset.items():
    atomic_query_responses[split_name] = {}
    split_data["target_text"] = split_data["target_text"].astype(str)
    for index, d in split_data.iterrows():
        rel = d["prefix"]
        if len(str(d["target_text"]))>0: 
            rel_token = atomic_relation_mappings[rel]
            event = d["input_text"]
            query = f"{event} {rel_token} {gen_token}"
            if query not in atomic_query_responses[split_name]:
                atomic_query_responses[split_name][query] = []
            atomic_query_responses[split_name][query].append(d["target_text"])
            #didn't convert ___ to <blank>
            #didn't normalize to lowercase

#flatten
print("building flattened pairs")
atomic_flattened = {}
for split_name,queries_responses in atomic_query_responses.items():
    atomic_flattened[split_name] = []
    for query,responses in queries_responses.items():
        for response in responses:
             atomic_flattened[split_name].append((query,response))
#%% tokenizer & model
tokenizer = MT5TokenizerFast.from_pretrained(load_path)
model = MT5ForConditionalGeneration.from_pretrained(load_path)



#%%
results = []
device = 'cuda:0'
model = model.to(device)
for i,(query,responses) in enumerate(tqdm(atomic_query_responses[val_set].items())):
    if num_generations>0 and i>= num_generations:
        break
    inputs = tokenizer(query,return_tensors='pt').to(device=device)
    hyps = model.generate(**inputs,**generation_params)
    hyps = tokenizer.batch_decode(hyps,skip_special_tokens=True)
    results.append({
        "head":query,
        "gens":hyps,
        "tails":responses
    })
#%%
with open(os.path.join(load_path,f"{val_set}_gen.json"),'w') as f:
    json.dump(results,f,ensure_ascii=False,indent=2)
# %%
refs = {r['head']:r['tails'] for r in results}
hyps = {r['head']:r['gens'] for r in results}
from comet.evaluation.myeval import QGEvalCap
QGEval = QGEvalCap("",refs,hyps,{"rouge":True, "blue":True}, os.path.join(load_path,"eval_result.jsonl"))
score_dict, scores_dict = QGEval.evaluate()
# %%
