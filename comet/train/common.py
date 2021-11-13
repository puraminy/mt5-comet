from pathlib import Path
import datetime
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
from pytz import timezone
tehran = timezone('Asia/Tehran')
now = datetime.datetime.now(tehran)
now = now.strftime('%Y-%m-%d-%H:%M')
home = expanduser("~")
colab = not "ahmad" in home and not "pouramini" in home
if not colab: 
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

mlog.info(now)

for logger, fname in zip([mlog,dlog,clog,vlog,tlog], ["all_main","all_data","all_cfg","all_eval","all_train"]):
    logger.setLevel(logging.INFO)
    logFilename = os.path.join(logPath, fname + ".log")
    handler = logging.FileHandler(logFilename, mode="w")
    logger.addHandler(handler)

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
def set_prompt_lengths(rel, enc_plen=0, dec_plen=0):
    global atomic_relation_prompt_lengths
    if rel == "":
        return
    atomic_relation_prompt_lengths[rel] = (enc_plen, dec_plen)


def extend_tokenizer(tokenizer, rel=""):
    if not rel:
        added_tokens = [ 
            AddedToken(token,lstrip=True,
                rstrip=False)
            for token in 
                list(atomic_relation_mappings.values())+
                list(gen_tokens.values())
        ]
        tokenizer.add_special_tokens({"additional_special_tokens":added_tokens}) 
    if rel:
        added_tokens = [ 
                AddedToken(prompt,lstrip=True,
                    rstrip=False)
                for prompt in encoder_prompts[rel] + decoder_prompts[rel]
        ]
        tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})

def wrap_model(model, tokenizer, rel, emb=False, prompt_path=""):
    id_offset = len(tokenizer)
    embedding_dim = model.config.hidden_size
    enc_plen = atomic_relation_prompt_lengths[rel][0]
    dec_plen = atomic_relation_prompt_lengths[rel][1]
    assert rel in encoder_prompts and len(encoder_prompts[rel]) > 0, "No encoder prompt defined!"
    dec_offset = id_offset + len(encoder_prompts[rel])
    prompt_encoder = None
    decoder_prompt_encoder = None
    mlog.info("id_offset: %s", id_offset)
    mlog.info("enc_plan: %s", enc_plen)
    mlog.info("dec_plan: %s", dec_plen)
    mlog.info("decoder offset: %s", dec_offset)
    if emb:
        if enc_plen > 0:
            prompt_encoder = EmbeddingPromptEncoder(enc_plen,embedding_dim,id_offset)
        if dec_plen > 0:
            decoder_prompt_encoder = EmbeddingPromptEncoder(dec_plen,embedding_dim,dec_offset)
    else:
        if enc_plen > 0:
            prompt_encoder = LSTMEmbeddingPromptEncoder(enc_plen,embedding_dim,id_offset)
        if dec_plen > 0:
            decoder_prompt_encoder = LSTMEmbeddingPromptEncoder(dec_plen,embedding_dim,dec_offset)

    extend_tokenizer(tokenizer, rel)
    model.resize_token_embeddings(len(tokenizer))
    wrapped_model = PTuningWrapper(model,prompt_encoder,decoder_prompt_encoder,prompt_token_fn=get_prompt_token_fn(id_offset,enc_plen + dec_plen))
    if prompt_path:
        mlog.info("Loading saved prompt...")
        wrapped_model.prompt_encoder.load(prompt_path)

    return wrapped_model

encoder_prompts = {} 
decoder_prompts = {}
def fill_consts(template, row):
    text = template
    dlog.info("Fill constants for %s", text)
    rel = row["prefix"]
    rel_token = atomic_relation_mappings[rel]        

    rel_natural_en = relation_natural_mappings[rel]["en"]        
    rel_natural_fa = relation_natural_mappings[rel]["fa"]        
    rep  = {"{rel}":rel, 
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
    enc_plen,dec_plen = atomic_relation_prompt_lengths[rel]
    if not rel in encoder_prompts:
        encoder_prompts[rel] = []
    if not rel in decoder_prompts:
        decoder_prompts[rel] = []
    counter = 0
    while "{enc_token}" in text:
        prompt = " ".join(f"<enc_{rel}_{i}>" for i in range(counter, enc_plen))
        if not prompt in encoder_prompts:
            encoder_prompts[rel].append(prompt)
        text = text.replace("{enc_token}",prompt, 1)
        counter += enc_plen 
    counter = 0
    while "{dec_token}" in text:
        prompt = " ".join(f"<dec_{rel}_{i}>" for i in range(counter, dec_plen))
        if not prompt in decoder_prompts:
            decoder_prompts[rel].append(prompt)
        text = text.replace("{dec_token}",prompt, 1)
        counter += dec_plen 
    for key,value in row.items():
        val = str(value)
        text = text.replace("{" + key + "}", val)

    dlog.info("After: %s", text)
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
def create_templates(method, wrapped, frozen, 
        gen_pos="end", prompt_pos="start", zero_shot=False, lang="mix"):
       if method == "pred-enfa":
           qtemp = "{enc_token_start} {gen_start} {input_text} {rel_natural_en} {gen_en} {ph} {event} {rel_natural} {enc_token_end} {gen_end} <extra_id_1>"
           anstemp = "{ph} {target_text} <extra_id_1> {resp} <extra_id_2>"
       elif method == "context-en":
           qtemp = "{enc_token_start} {gen_start} {input_text} {rel_natural_en} {gen_en} {target_text} {enc_token_start} {event} {rel_natural} {enc_token_end} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "context-faen":
           qtemp = "{enc_token_start} {gen_start} {input_text_fa} {rel_natural_fa} {gen_en} {target_text} {event} {rel_natural} {enc_token_end} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "context-enfa":
           qtemp = "{enc_token_start} {gen_start} {input_text} {rel_natural_en} {gen_fa} {target_text_fa} {enc_token_start} {event} {rel_natural} {enc_token_end} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "context-fa":
           qtemp = "{enc_token_start} {gen_start} {input_text_fa} {rel_natural_fa} {gen_fa} {target_text_fa} {enc_token_start} {event} {rel_natural} {enc_token_end} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "sup":
           qtemp = "{enc_token_start} {gen_start} {event} {enc_token_end} {gen_end}"
           anstemp = "{resp}"
       elif method == "unsup":
           qtemp = "{enc_token_start} {gen_start} {event} {enc_token_end} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup-dec":
           qtemp = "{enc_token_start} {gen_start} {event} {enc_token_end} {gen_end} {ph}"
           anstemp = "{dec_token} {ph} {resp} {end}"
       elif method == "unsup-2":
           qtemp = "{enc_token} {gen_start} {event} {enc_token} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       else:
           raise ValueError("not supprted method: " + method)
       if gen_pos == "end":
           qtemp = qtemp.replace("{gen_start} ","")
           qtemp = qtemp.replace("{gen_end}","{gen}")
       else:
           qtemp = qtemp.replace(" {gen_end}","")
           qtemp = qtemp.replace("{gen_start}","{gen}")
       if prompt_pos == "end":
           qtemp = qtemp.replace("{enc_token_start} ","")
           qtemp = qtemp.replace("{enc_token_end}","{enc_token}")
       else:
           qtemp = qtemp.replace(" {enc_token_end} ","")
           qtemp = qtemp.replace("{enc_token_start}","{enc_token}")
       if not wrapped:
           qtemp = qtemp.replace("{enc_token}","{rel_token}")
       if zero_shot:
           qtemp = qtemp.replace("{rel_token}","")
       if lang != "mix":
           qtemp = qtemp.replace("{gen}","")
           qtemp = qtemp.replace("{gen_en}","")
           qtemp = qtemp.replace("{gen_fa}","")
       while "  " in qtemp:
           qtemp = qtemp.replace("  "," ")


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
            nli_group="all", is_record=False, start=0): 
    dlog.info("building query responses for {}".format(split_name))
    dlog.info(f"len:{len(split_df)}")
    dlog.info(f"qtemp:{qtemp}")
    dlog.info(f"anstemp:{anstemp}")
    natural = include == "natural"
    if split_name != "train":
        start = 0
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

    cat_counter = {}
    ii = 0
    kk = 0
    dlog.info(f"len after filtering:{len(split_df)}")
    flat_data = []
    old_input = ""
    for index, d in split_df.iterrows():
        rel = d["prefix"]
        ii += 1
        if ii < start:
            continue
        _qtemp = fill_consts(qtemp,d)
        _anstemp = fill_consts(anstemp,d)
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
                query = fill_vars(_qtemp, rel, event, gen_token, resp, 
                        input_lang, target_lang) 
                response = fill_vars(_anstemp, rel, event, gen_token, resp, 
                        input_lang, target_lang)
                lang = input_lang + "2" + target_lang
                if not lang in cat_counter:
                    cat_counter[lang] = 1
                else:
                    cat_counter[lang] += 1
                if cat_counter[lang] < 3:
                    dlog.info(f"%%%%%%%%%%%%%%%%%% {lang} %%%%%%%%%%%%%%%%%%%")
                    dlog.info(inp + "====>" + targ_col)
                    dlog.info(input_lang + ":"+ query)
                    dlog.info(target_lang + ":" + response)
                if cat_counter[lang] > num_samples:
                    return data_split, flat_data, kk
                if not lang in data_split[rel]:
                    data_split[rel][lang] = []
                if query not in data_split[rel][lang]:
                    data_split[rel][lang].append({query:[response]})
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
    if save_path.endswith("temp"):
        mlog.info("Saves in temp are skipped ...")
        return

    mlog.info("Saving model ...")
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

