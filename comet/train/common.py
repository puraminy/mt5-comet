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

SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "</s>",
                    "unk_token": "<|UNK|>",
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}
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
    "oReact":{ 
        "en":"as a result others feel ",
        "fa":"در نتیجه دیگران حس می کنند"
    },
    "xReact":{ 
        "en":"as a result PersonX feels ",
        "fa":"در نتیجه PersonX حس می کند", 
    },
    "xWant":{ 
        "en":"Then PersonX wants ",
        "fa":"بعد از آن PersonX می خواهد"
    },
    "oWant":{ 
        "en":"Then others want ",
        "fa":"بعد از آن دیگران می خواهند"
    },
    "xEffect":{ 
        "en":"as a result PersonX  ",
        "fa":"در نتیجه PersonX "
    },
    "oEffect":{ 
        "en":"as a result others  ",
        "fa":"در نتیجه دیگران "
    },
    "xAttr":{ 
        "en":"PersonX is seen as",
        "fa":"مردم فکر می کنند PersonX "
    },
    "xIntent":{ 
        "en":"because PersonX intended ",
        "fa":"زیرا PersonX می خواست"
    },
    "xNeed":{ 
        "en":"Before that, PersonX needs ",
        "fa":"قبل از آن PersonX نیاز دارد"
    },
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
end_token = SPECIAL_TOKENS['eos_token']  #"</s>"
# %%
atomic_relation_prompt_lengths = {
    "xAttr":[5,3],
    "xEffect":[5,3],
    "oEffect":[5,3],
    "xReact":[5,3],
    "oReact":[5,3],
    "xWant":[5,3],
    "oWant":[5,3],
    "xIntent":[5,3],
    "xNeed":[5,3],
}

def get_prompt_token_fn(id_offset,length):
    return lambda x: (x>=id_offset)&(x<=id_offset+length+1)

encoder_relation_mappings = {}
decoder_relation_mappings = {}
def set_prompt_lengths(rel, length):
    if rel == "":
        return
    atomic_relation_prompt_lengths[rel] = length


def extend_tokenizer(tokenizer, rel=""):
    if not rel:
        added_tokens = [ 
            AddedToken(token,lstrip=True,
                rstrip=False)
            for token in 
                list(atomic_relation_mappings.values())+
                list(gen_tokens.values())
        ]
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
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
    enc_plen = len(encoder_prompts[rel])
    dec_plen = len(decoder_prompts[rel])
    assert rel in encoder_prompts and enc_plen > 0, "No encoder prompt defined!"
    dec_offset = id_offset + enc_plen
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
        if Path(os.path.join(prompt_path, "encoder")).exists():
            mlog.info("Loading saved encoder prompt...")
            wrapped_model.prompt_encoder.load(prompt_path)
        if Path(os.path.join(prompt_path, "decoder")).exists():
            mlog.info("Loading saved decoder prompt...")
            wrapped_model.decoder_prompt_encoder.load(prompt_path)

    return wrapped_model

encoder_prompts = {} 
decoder_prompts = {}
def fill_consts(template, extemp, row, rows=[]):
    text = template
    #dlog.debug("fill const for: %s", text)
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
    for key,value in row.items():
        val = str(value)
        text = text.replace("{" + key + "}", val)

    plen = atomic_relation_prompt_lengths[rel]
    if not rel in encoder_prompts:
        encoder_prompts[rel] = []
    if not rel in decoder_prompts:
        decoder_prompts[rel] = []
    counter = 0
    pi = 0
    enc_prompt = ""
    dec_prompt = ""
    while "{enc_token}" in text:
        enc_plen = plen[pi] if pi < len(plen) else plen[-1] 
        prompt = ""
        for i in range(counter, counter + enc_plen):
            token = f"<enc_{rel}_{i}>" 
            prompt += " " + token
            if not token in encoder_prompts[rel]:
                encoder_prompts[rel].append(token)
        prompt = prompt.strip()
        if not enc_prompt:
            enc_prompt = prompt
        text = text.replace("{enc_token}",prompt, 1)
        counter += enc_plen 
        pi += 1
    if "{examples}" in text:
        examples = ""
        ii = 1
        for idx, _row in rows.iterrows():
            example = extemp
            if "{enc_token}" in extemp:
                assert enc_prompt != "", "Prompt was not set!"
            example = pattern.sub(lambda m: rep[re.escape(m.group(0))], example)
            example = example.replace("{enc_token}", enc_prompt)
            example = example.replace("{dec_token}", dec_prompt)
            for key,value in _row.items():
                val = str(value)
                example = example.replace("{" + key + "}", val)
            examples += " " + str(ii) + ") " + example
            ii += 1

        text = text.replace("{examples}", examples + " " + str(ii) + ")")

    #dlog.debug("after: %s", text)
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
       extemp = ""
       if method == "sup-pred-enfa":
           qtemp = "{input_text} {enc_token} {gen_fa}"
           anstemp = "{input_text_fa} {dec_token} {target_text_fa}"
       elif method == "sup-enfa":
           qtemp = "{input_text} {enc_token} {target_text} {gen_fa}"
           anstemp = "{input_text_fa} {dec_token} {target_text_fa}"
       elif method == "sup-enmix":
           qtemp = "{input_text} {enc_token} {target_text} {gen}"
           anstemp = "{event} {dec_token} {gen} {resp}"
       elif method == "unsup-nat":
           qtemp = "{enc_token} {event} {rel_natural} {gen} {ph}" 
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup-no-gen":
           qtemp = "{enc_token} {event} {enc_lang_token} {ph}" 
           anstemp = "{ph} {resp} {end}"
       elif method == "sup-gen":
           qtemp = "{event} {gen}"
           anstemp = "{resp} {end}"
       elif method == "sup-no-gen":
           qtemp = "{event}"
           anstemp = "{resp} {end}"
       elif method == "gen":
           qtemp = "{gen}"
           anstemp = "{resp}"
       elif method == "pred-enfa":
           qtemp = "{enc_token_start} {gen_start} {input_text} {rel_natural_en} {gen_en} {ph} {event} {rel_natural} {enc_token_end} {gen_end} <extra_id_1>"
           anstemp = "{ph} {target_text} <extra_id_1> {resp} <extra_id_2>"
       elif method == "context-en":
           qtemp = "{enc_token_start} {gen_start} {input_text} {rel_natural_en} {gen_en} {target_text} {enc_token_start} {event} {rel_natural} {enc_token_end} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "context-faen":
           qtemp = "{enc_token_start} {gen_start} {input_text_fa} {rel_natural_fa} {gen_en} {target_text} {event} {rel_natural} {enc_token_end} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "context-n":
           qtemp = "{examples} {event} {enc_token} {gen} {ph}"
           extemp = "{input_text} {enc_token} {target_text} {end}"
           anstemp = "{ph} {resp} {end}"
       elif method == "trans":
           qtemp = "{target_text} en2fa"
           anstemp = "{target_text_fa}"
       elif method == "event-n":
           qtemp = "{examples} {gen} {ph}"
           extemp = "{gen} {input_text} {end} \n"
           anstemp = "{ph} {event} {end}"
       elif method == "gpt-event-n":
           qtemp = "{examples} {gen}"
           extemp = "{gen} {input_text} {end} \n"
           anstemp = "{event} {end}"
       elif method == "event-n-wrap":
           qtemp = "{examples} {enc_token} {ph}"
           extemp = "{enc_token} {input_text} {end} \n"
           anstemp = "{ph} {event} {end}"
       elif method == "gpt-n":
           qtemp = "{examples} {event} {rel_natural}"
           extemp = "{input_text} {rel_natural} {target_text} {end}"
           anstemp = "{resp} {end}"
       elif method == "gpt-n-wrap":
           qtemp = "{examples} {event} {enc_token}"
           extemp = "{input_text} {enc_token} {target_text} {end}"
           anstemp = "{resp} {end}"
       elif method == "context-n-dec":
           qtemp = "{event} {enc_token} {gen} {ph}"
           extemp = "{input_text} {target_text} {end}"
           anstemp = "{examples} {ph} {resp} {end}"
       elif method == "context-enfa":
           qtemp = "{enc_token_start} {gen_start} {input_text} {rel_natural_en} {gen_fa} {target_text_fa} {enc_token_start} {event} {rel_natural} {enc_token_end} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "context-fa":
           qtemp = "{enc_token_start} {gen_start} {input_text_fa} {rel_natural_fa} {gen_fa} {target_text_fa} {enc_token_start} {event} {rel_natural} {enc_token_end} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "sup":
           qtemp = "{enc_token_start} {gen_start} {event} {enc_token_end} {gen_end}"
           anstemp = "{resp} {end}"
       elif method == "gpt-wrap":
           qtemp = "{event} {enc_token}"
           anstemp = "{resp} {end}"
       elif method == "gpt":
           qtemp = "{event} {rel_natural}"
           anstemp = "{resp} {end}"
       elif method == "unsup":
           qtemp = "{enc_token_start} {gen_start} {event} {enc_token_end} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup-dec":
           qtemp = "{enc_token_start} {gen_start} {event} {enc_token_end} {gen_end} {ph}"
           anstemp = "{ph} {dec_token}{resp} {end}"
       elif method == "unsup-2":
           qtemp = "{enc_token} {gen_start} {event} {enc_token} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup-3":
           qtemp = "{enc_token} {gen_start} {event} {enc_token} {gen_end} {ph}"
           anstemp = "{ph} {dec_token} {resp} {end}"
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
           mlog.info("Not Wrapped")
           qtemp = qtemp.replace("{enc_token}","{rel_token}")
       if zero_shot:
           qtemp = qtemp.replace("{rel_token}","")
#       if not "mix" in lang:
#           qtemp = qtemp.replace("{gen}","")
#           qtemp = qtemp.replace("{gen_en}","")
#           qtemp = qtemp.replace("{gen_fa}","")
       while "  " in qtemp:
           qtemp = qtemp.replace("  "," ")


       return qtemp, anstemp, extemp

def fill_vars(template, rel, event, gen_token, resp, inp_lang, resp_lang):
    rel_natural = relation_natural_mappings[rel][inp_lang]        
    rep  = {"{event}":event, 
            "{resp}":resp,
            "{rel_natural}":rel_natural,
            "{gen}":gen_token}
    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], template)
    lang = resp_lang
    plen = atomic_relation_prompt_lengths[rel]
    if not rel in encoder_prompts:
        encoder_prompts[rel] = []
    if not rel in decoder_prompts:
        decoder_prompts[rel] = []
    counter = 0
    enc_prompt = ""
    pi = 0
    while "{enc_lang_token}" in text:
        enc_plen = plen[pi] if pi < len(plen) else plen[-1] 
        prompt = ""
        for i in range(counter, counter + enc_plen):
            token = f"<enc_{lang}_{rel}_{i}>" 
            prompt += " " + token
            if not token in encoder_prompts[rel]:
                encoder_prompts[rel].append(token)
        prompt = prompt.strip()
        if not enc_prompt:
            enc_prompt = prompt
        text = text.replace("{enc_lang_token}",prompt, 1)
        counter += enc_plen 
        pi += 1
    counter = 0
    dec_prompt = ""
    while "{dec_lang_token}" in text:
        dec_plen = plen[pi] if pi < len(plen) else plen[-1] 
        prompt=""
        for i in range(counter, counter+dec_plen):
            token = f"<dec_{lang}_{rel}_{i}>" 
            prompt += " " + token
            if not token in decoder_prompts[rel]:
                decoder_prompts[rel].append(token)
        prompt = prompt.strip()
        if not dec_prompt:
            dec_prompt = prompt
        text = text.replace("{dec_lang_token}",prompt, 1)
        counter += dec_plen 
        pi += 1
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
def fill_data(split_df, split_name, qtemp, anstemp, extemp, 
            num_samples=0, 
            ignore_blanks=False,
            inp_include="",
            inp_exclude="",
            targ_include="",
            targ_exclude="",
            pred_tresh=0,
            nli_group="all", is_record=False, start=0, sampling=0, samples_per_head=2): 
    dlog.info("building query responses for {}".format(split_name))
    dlog.info(f"len:{len(split_df)}")
    dlog.info(f"qtemp:{qtemp}")
    dlog.info(f"anstemp:{anstemp}")
    natural = inp_include == "natural"
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
    si = 0
    for index, d in split_df.iterrows():
        rel = d["prefix"]
        ii += 1
        eng_inp = d["input_text"]
        si += 1
        if eng_inp != old_input:
            old_input = eng_inp
            si = 0
        elif si > samples_per_head:
            continue
        if ii < start:
            continue
        context_rows=[]
        if sampling > 0:
            context_rows = split_df.sample(n=sampling)
        _qtemp = fill_consts(qtemp, extemp,d, context_rows)
        _anstemp = fill_consts(anstemp, extemp,d, context_rows)
        if not rel in data_split:
            data_split[rel] = {}
        for inp in inputs:
            if not inp in d or len(d[inp]) <= 1:
                continue
            if inp_include and not any(x in inp for x in inp_include.split("|")):
                continue
            if inp_exclude and any(x in inp for x in inp_exclude.split("|")):
                continue
            input_lang = langs[inp]
            for targ_col in targets:
                if not targ_col in d or len(d[targ_col]) <= 1:
                    continue
                if targ_include and not any(x in targ_col for x in targ_include.split("|")):
                    continue
                if targ_exclude and any(x in targ_col for x in targ_exclude.split("|")):
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
                _query = fill_vars(_qtemp, rel, event, gen_token, resp, 
                        input_lang, target_lang) 
                query = (index, _query)
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
                    dlog.info(input_lang + ":"+ _query)
                    dlog.info(target_lang + ":" + response)
                if cat_counter[lang] > num_samples:
                    return data_split, flat_data, kk
                if not lang in data_split[rel]:
                    data_split[rel][lang] = []
                if query not in data_split[rel][lang]:
                    data_split[rel][lang].append({query:[response]})
                else:
                    data_split[rel][lang][query].append(response)
                flat_data.append((_query, response))
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

