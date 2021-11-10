from pathlib import Path
from transformers import AddedToken 
import pandas as pd
from comet.transformers_ptuning import PTuningWrapper
from comet.transformers_ptuning.ptuning_wrapper import LSTMEmbeddingPromptEncoder, EmbeddingPromptEncoder
from tqdm import tqdm
import logging, sys
import re
import os
import torch
import json
from os.path import expanduser
home = expanduser("~")
if "ahmad" in home or "pouramini" in home:
    logPath = os.path.join(home, "logs")
    resPath = os.path.join(home, "results") 
else:
    home = "/content/drive/MyDrive/pouramini"
    logPath = "/content/"
    resPath = "/content/drive/MyDrive/pouramini/results"

Path(resPath).mkdir(exist_ok=True, parents=True)
Path(logPath).mkdir(exist_ok=True, parents=True)

logFilename = os.path.join(logPath, "all.log") #app_path + '/log_file.log'
# log only important messages
logging.basicConfig(filename=logFilename)
consoleHandler = logging.StreamHandler()
mlog = logging.getLogger("comet.main")
mlog.setLevel(logging.INFO)
mlog.addHandler(consoleHandler)
clog = logging.getLogger("comet.cfg")
dlog = logging.getLogger("comet.data")
vlog = logging.getLogger("comet.eval")
tlog = logging.getLogger("comet.train")


for logger, fname in zip([mlog,dlog,clog,vlog,tlog], ["all_main","all_data","all_cfg","all_eval","all_train"]):
    logger.setLevel(logging.INFO)
    logFilename = os.path.join(logPath, fname + ".log")
    handler = logging.FileHandler(logFilename, mode="w")
    logger.addHandler(handler)


device = 'cuda'
def set_device(dev):
    global device
    device = dev

results = {}
resFile = os.path.join(resPath, "results.json")
if Path(resFile).exists():
    with open(resFile, "r") as f:
        mlog.info("Reading stored results...")
        results = json.load(f)

nli_map = ['contradiction', 'entailment', 'neutral']
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
relation_natural_mappings = {
    "oEffect":"<oEffect>",
    "oReact":"<oReact>",
    "oWant":"<oWant>",
    "xAttr":"<xAttr>",
    "xEffect":"<xEffect>",
    "xIntent":{ 
        "en":"because PersonX intended ",
        "fa":"زیرا PersonX می خواست"
    },
    "xNeed":"<xNeed>",
    "xReact":"<xReact>",
    "xWant":"<xWant>"
}
gen_token_en = "<gen_en>"
gen_token_fa = "<gen_fa>"
gen_tokens = {"target_text":gen_token_en, 
              "target_text_fa":gen_token_fa,
              "natural_target_text_fa":gen_token_fa,
              "natural_target_text":gen_token_en,
              "natural_input_text_fa":gen_token_fa,
              "natural_input_text":gen_token_en,
              "pred_text1":gen_token_en,
              "pred_text_fa":gen_token_fa,
              "all_preds":gen_token_en,
              "en2en":gen_token_en,
              "fa2en":gen_token_en,
              "en2fa":gen_token_fa,
              "fa2fa":gen_token_fa,
              "all_preds_fa":gen_token_fa}
langs = {"target_text":"en", 
              "target_text_fa":"fa",
              "input_text":"en",
              "input_text_fa":"fa",
              "natural_target_text_fa":"fa",
              "natural_target_text":"en",
              "natural_input_text_fa":"fa",
              "natural_input_text":"en",
              "pred_text1":"en",
              "pred_text_fa":"fa",
              "all_preds":"en",
              "all_preds_fa":"fa"}
              
targets = ["target_text", "target_text_fa", "pred_text1", "all_preds", "pred_text_fa","all_preds_fa", "natural_target_text_fa", "natural_target_text"]
inputs = ["input_text", "input_text_fa", "natural_input_text", "natural_input_text_fa"]

placeholder_token = "<extra_id_0>"
end_token = "<extra_id_1>"
# %%
atomic_relation_prompt_lengths = {
    "xIntent":(5,3),
}

def get_prompt_token_fn(id_offset,length):
    return lambda x: (x>=id_offset)&(x<=id_offset+length+1)

encoder_relation_mappings = {}
decoder_relation_mappings = {}
def map_relations():
    global encoder_relation_mappings, decoder_relation_mappings
    for rel,(enc_plen, dec_plen) in atomic_relation_prompt_lengths.items():
        encoder_relation_mappings[rel] = " ".join(f"<{rel}_{i}>" for i in range(enc_plen))
        decoder_relation_mappings[rel] = " ".join(f"<{rel}_{i}>" for i in range(enc_plen,enc_plen+dec_plen))
    return encoder_relation_mappings, decoder_relation_mappings

def extend_tokenizer(tokenizer, rel=""):
    added_tokens = [ 
        AddedToken(token,lstrip=True,
            rstrip=False)
        for token in 
            list(atomic_relation_mappings.values())+
            list(gen_tokens.values())
    ]
    tokenizer.add_special_tokens({"additional_special_tokens":added_tokens}) 
    if rel:
        enc_plen = atomic_relation_prompt_lengths[rel][0]
        dec_plen = atomic_relation_prompt_lengths[rel][1]
        added_tokens = [ 
                AddedToken(f"<{rel}_{i}>",lstrip=True,
                    rstrip=False)
                for i in 
                    range(enc_plen + dec_plen + 1)
        ]
        tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
    return tokenizer

def wrap_model(model, tokenizer, rel, emb=False, prompt_path=""):
    id_offset = len(tokenizer)
    embedding_dim = model.config.hidden_size
    enc_plen = atomic_relation_prompt_lengths[rel][0]
    dec_plen = atomic_relation_prompt_lengths[rel][1]
    dec_offset = id_offset + enc_plen
    prompt_encoder = None
    decoder_prompt_encoder = None
    if emb:
        prompt_encoder = EmbeddingPromptEncoder(enc_plen,embedding_dim,id_offset)
        decoder_prompt_encoder = EmbeddingPromptEncoder(dec_plen,embedding_dim,dec_offset)
    else:
        prompt_encoder = LSTMEmbeddingPromptEncoder(enc_plen,embedding_dim,id_offset)
        decoder_prompt_encoder = LSTMEmbeddingPromptEncoder(dec_plen,embedding_dim,dec_offset)

    #tokenizer = extend_tokenizer(tokenizer, rel)
    #model.resize_token_embeddings(len(tokenizer))
    wrapped_model = PTuningWrapper(model,prompt_encoder,decoder_prompt_encoder,prompt_token_fn=get_prompt_token_fn(id_offset,enc_plen + dec_plen))
    if prompt_path:
        wrapped_model.prompt_encoder.load(prompt_path)

    return wrapped_model

def fill_consts(template, row):
    text = template
    rel = row["prefix"]
    rel_token = atomic_relation_mappings[rel]        
    enc_token = encoder_relation_mappings[rel] if rel in encoder_relation_mappings else ""
    dec_token = decoder_relation_mappings[rel] if rel in encoder_relation_mappings else ""
    rel_natural_en = relation_natural_mappings[rel]["en"]        
    rel_natural_fa = relation_natural_mappings[rel]["fa"]        
    rep  = {"{rel}":rel, 
            "{enc_token}":enc_token, 
            "{dec_token}":dec_token, 
            "{rel_token}":rel_token,
            "{rel_natural_en}":rel_natural_en,
            "{rel_natural_fa}":rel_natural_fa,
            "{gen_fa}":gen_token_fa,
            "{gen_en}":gen_token_en,
            "{ph}":placeholder_token,
            "{end}":end_token}
    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], template)
    for key,value in row.items():
        val = str(value)
        text = text.replace("{" + key + "}", val)
    return text

def filter_inputs(include, exclude, lang):
   if lang == "en":
       exclude += "|natural|_fa"
   if lang == "fa":
       exclude += "|natural"
       include += "|_fa"
   if lang == "mix":
       exclude += "|natural"
   include = include.strip("|")
   exclude = exclude.strip("|")
   return include, exclude

#tttttttttt
def create_templates(method, wrapped, frozen):
       if method == "context-en":
           qtemp = "{enc_token} {input_text} {rel_natural_en} {target_text} {event} {rel_natural} {gen} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "context-fa":
           qtemp = "{enc_token} {input_text_fa} {rel_natural_fa} {target_text_fa} {event} {rel_natural} {gen} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "sup":
           if wrapped:
               qtemp = "{enc_token} {event} {gen}"
               anstemp = "{resp}"
           else: #unwrapped
               qtemp = "{rel_token} {event} {gen}"
               anstemp = "{resp}"
       elif method == "unsup":
           if wrapped:
               qtemp = "{enc_token} {event} {gen} {ph}"
               anstemp = "{ph} {resp} {end}"
           else: #unwrapped
               if frozen:
                   qtemp = "{rel_token} {event} {gen} {ph}"
                   anstemp = "{ph} {resp} {end}"
               else:
                   qtemp = "{event} {rel_natural} {ph}"
                   anstemp = "{ph} {resp} {end}"
       else:
           raise ValueError("not supprted method: " + method)
       return qtemp, anstemp

def fill_vars(template, rel, event, gen_token, resp, inp_lang, resp_lang):
    rel_natural = relation_natural_mappings[rel][inp_lang]        
    rep  = {"{event}":event, 
            "{resp}":resp,
            "{rel_natural}":rel_natural,
            "{gen}":gen_token}
    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], template)
    return text

def get_input(msg):
    while True:
        try:
            ans = input(msg)
            break
        except KeyboardInterrupt:
            break
        except:
            print("Error occured, please try again or hit e to exit")
            continue
# fill a dataset or generate based on a model
# mmmmmmmmmmmmmm
def fill_data(split_df, split_name, qtemp, anstemp, 
            num_samples=0, 
            ignore_blanks=False,
            include="",
            exclude="",
            pred_tresh=0,
            nli_group="all", is_record): 
    dlog.info("building query responses for {}".format(split_name))
    dlog.info(f"len:{len(split_df)}")
    dlog.info(f"qtemp:{qtemp}")
    dlog.info(f"anstemp:{anstemp}")
    natural = include == "natural"
    if natural and split_name != "train": natural = False 
    if natural:
        dlog.info("natural is ON")
    data_split = {}
    if num_samples == 0: num_samples = len(split_df)
    split_df = split_df.sort_values(by="input_text")
    for col in targets:
        if col in split_df:
            split_df[col] = split_df[col].astype(str)
    if ignore_blanks: # and len(split_df) > num_rows:
        split_df = split_df[split_df["input_text"].str.contains('___')==False]
    if pred_tresh > 0 and "bert_score" in split_df:
        split_df = split_df[split_df["bert_score"] > pred_tresh]
        dlog.info("*** Filtered based on pred1 score higher than "+ pred_tresh)
    if nli_group != "all" and "nli_group" in split_df:
        split_df = split_df[split_df["nli_group"] == nli_group]
        dlog.info("*** Filtered based on nli_group "+ nli_group)

    jj = 0
    ii = 0
    kk = 0
    dlog.info(f"len after filtering:{len(split_df)}")
    flat_data = []
    old_input = ""
    pbar = tqdm(total = num_samples)
    for index, d in split_df.iterrows():
        rel = d["prefix"]
        query = fill_consts(qtemp,d)
        response = fill_consts(anstemp,d)
        if not rel in data_split:
            data_split = {rel:{}}
        for inp in inputs:
            if not inp in d or len(d[inp]) <= 1:
                continue
            if include and not any(x in inp for x in include.split("|")):
                continue
            if exclude and any(x in inp for x in exclude.split("|")):
                continue
            input_lang = langs[inp]
            for targ_col in targets:
                if not targ_col in d or len(d[targ_col]) <= 1:
                    continue
                if include and not any(x in targ_col for x in include.split("|")):
                    continue
                if exclude and any(x in targ_col for x in exclude.split("|")):
                    continue
                rel_token = atomic_relation_mappings[rel]
                event = d[inp]
                resp = d[targ_col]
                if natural:
                    resp = resp.replace("PersonX intends", "")
                    resp = resp.replace("PersonX قصد دارد", "")
                resp = resp.strip()
                gen_token = gen_tokens[targ_col]
                target_lang = langs[targ_col]
                query = fill_vars(query, rel, event, gen_token, resp, 
                        input_lang, target_lang) 
                response = fill_vars(response, rel, event, gen_token, resp, 
                        input_lang, target_lang)
                lang = input_lang + "2" + target_lang
                if not lang in data_split[rel]:
                    data_split[rel][lang] = []
                if d["input_text"] != old_input:
                    old_input = d["input_text"]
                    ii+=1
                if ii >= num_samples:
                    return data_split, flat_data, kk
                if query not in data_split[rel][lang]:
                    jj+=1
                    pbar.update(1)
                    data_split[rel][lang].append({query:[response]})
                    if jj < 3:
                        dlog.info("Q:"+ query)
                        dlog.info("R:"+ response)
                else:
                    data_split[rel][lang][query].append(response)
                flat_data.append((query, response))
                kk += 1
                if is_record and kk > num_samples:
                    return data_split, flat_data, kk
            #didn't convert ___ to <blank>
            #didn't normalize to lowercase
    return data_split, flat_data, kk

def save_checkpoint(model, optimizer, scheduler, step, 
                   best_eval_step, best_dev_loss, save_path):
    mlog.info("Saving model ...")
    return
    with open(save_path + "/best_model.txt", "a") as f:
        print("best_step:", best_eval_step, file=f)
        print("best dev loss:", best_dev_loss, file=f)
    model.save_pretrained(save_path)

#    torch.save({
#            'step': step,
#            'eval_step': best_eval_step,
#            'model_state_dict': model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'scheduler_state_dict': scheduler.state_dict(),
#            }, os.path.join(save_path, "saved_states"))
#

