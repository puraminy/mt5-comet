from pathlib import Path

import math
from termcolor import colored
from transformers import AddedToken 
from functools import lru_cache
import pandas as pd
import numpy as np
#import tensorflow as tf
from comet.utils.myutils import *
from comet.transformers_ptuning import PTuningWrapper
from comet.transformers_ptuning.ptuning_wrapper import * 
from comet.polytropon import SkilledMixin
from tqdm import tqdm
import logging, sys
import random
import comet.train.tokens as tokens
import re
import os
import torch
import json
from comet.train.mylogs import *
import pickle5 as pickle


main_args = {}
def set_args(args):
    global main_args 
    main_args =args

SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "</s>",
                    "unk_token": "<|UNK|>",
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}

sep = "<|SEP|>"
pad_token = {"pad_token": "<|PAD|>"}
sep_token = {"sep_token": sep}
nli_map = ['contradiction', 'entailment', 'neutral']
rel_maps = {
    "cb":"<cb>",
    "oEffect":"<oEffect>",
    "oReact":"<oReact>",
    "oWant":"<oWant>",
    "xAttr":"<xAttr>",
    "xEffect":"<xEffect>",
    "xIntent":"<xIntent>",
    "xNeed":"<xNeed>",
    "xReact":"<xReact>",
    "xWant":"<xWant>",
    'AtLocation':"<loc>", 
    'ObjectUse':"<use>", 
    'Desires': "<desire>",
    'HasProperty':"<prop>", 
    'NotDesires': "<no_desire>", 
    'Causes':"<cause>", 
    'HasSubEvent':"<has_sub>", 
    'xReason':"<xReason>",
    'CapableOf':"<capable>", 
    'MadeUpOf':"<madeof>", 
    'isAfter':"<isAfter>", 
    'isBefore':"<isBefore>", 
    'isFilledBy': "<isFilledBy>",
    'HinderedBy':"<HinderedBy>"
}
# omit some words from target for better rouge score calculation
rel_target_omits = {
    "xIntent":"to",
}
all_rels = [key for key,val in rel_maps.items()] 
x_rels = [key for key,val in rel_maps.items()]
rel_nat_maps = {
    "cb":{ 
        1:"{event1}? {ph}. {event2}",
        2:"{event1}? {ph}. {event2}",
        3:"sentence1: {event1}. sentence2: {event2}. The relation is {ph}.",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"the entailment is",
        "fa":"رابطه میان دو جمله",
        },
    "xAttr":{ 
        1:"{event}, So PersonX is seen as {ph}.",
        2:"{event}, So PersonX is seen as {ph}.",
        3:"{event}, PersonX is seen as {ph}.",
        4:"PersonX is seen as {ph} because {event}",
        "rel_qtemp": "{event}. {rel_5} PersonX is seen as {ph}",
        "rel_anstemp":"{ph} {resp} {end}",
        "fa":"مردم فکر می کنند PersonX ",
        "tokens":"<state> <agent> <static>",
        "n-tokens":"event {event}, agent state static is {ph}",
        "nat-tokens":"always, the state of the person is",
        "desc":"the person's attributes"
    },
    "AtLocation":{ 
        1:"{event} is located in {ph}.",
        2:"{event} is found on {ph}",
        "fa":"در نتیجه دیگران حس می کنند",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"then, the state of others is ",
        "desc":"other's reaction to the event"
    },
    "ObjectUse":{ 
        1:"{event} is used for {ph}.",
        2:"Others feel {ph} because {event}",
        "fa":"در نتیجه دیگران حس می کنند",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"then, the state of others is ",
        "desc":"other's reaction to the event"
    },
    "Desires":{ 
        1:"{event} desires {ph}.",
        2:"Others feel {ph} because {event}",
        "fa":"در نتیجه دیگران حس می کنند",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"then, the state of others is ",
        "desc":"other's reaction to the event"
    },
    "HasProperty":{ 
        1:"{event} can be characterized by having {ph}.",
        2:"Others feel {ph} because {event}",
        "fa":"در نتیجه دیگران حس می کنند",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"then, the state of others is ",
        "desc":"other's reaction to the event"
    },
    "NotDesires":{ 
        1:"{event} does not desire {ph}.",
        2:"Others feel {ph} because {event}",
        "fa":"در نتیجه دیگران حس می کنند",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"then, the state of others is ",
        "desc":"other's reaction to the event"
    },
    "Causes":{ 
        1:"{event} causes {ph}.",
        2:"Others feel {ph} because {event}",
        "fa":"در نتیجه دیگران حس می کنند",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"then, the state of others is ",
        "desc":"other's reaction to the event"
    },
    "HasSubEvent":{ 
        1:"{event} includes {ph}",
        2:"Others feel {ph} because {event}",
        "fa":"در نتیجه دیگران حس می کنند",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"then, the state of others is ",
        "desc":"other's reaction to the event"
    },
    "xReason":{ 
        1:"{event} because {ph}",
        2:"Others feel {ph} because {event}",
        "fa":"در نتیجه دیگران حس می کنند",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"then, the state of others is ",
        "desc":"other's reaction to the event"
    },
    "CapableOf":{ 
        1:"{event} is capable of {ph}",
        2:"Others feel {ph} because {event}",
        "fa":"در نتیجه دیگران حس می کنند",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"then, the state of others is ",
        "desc":"other's reaction to the event"
    },
    "MadeUpOf":{ 
        1:"{event} is made up of {ph}",
        2:"Others feel {ph} because {event}",
        "fa":"در نتیجه دیگران حس می کنند",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"then, the state of others is ",
        "desc":"other's reaction to the event"
    },
    "isAfter":{ 
        1:"{event} happens after {ph}",
        2:"Others feel {ph} because {event}",
        "fa":"در نتیجه دیگران حس می کنند",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"then, the state of others is ",
        "desc":"other's reaction to the event"
    },
    "isBefore":{ 
        1:"{event} happens before {ph}",
        2:"Others feel {ph} because {event}",
        "fa":"در نتیجه دیگران حس می کنند",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"then, the state of others is ",
        "desc":"other's reaction to the event"
    },
    "isFilledBy":{ 
        1:"{event} can be filled by {ph}",
        2:"{event}",
        "fa":"در نتیجه دیگران حس می کنند",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"then, the state of others is ",
        "desc":"other's reaction to the event"
    },
    "HinderedBy":{ 
        1:"{event} can be hindered by {ph}",
        2:"Others feel {ph} because {event}",
        "fa":"در نتیجه دیگران حس می کنند",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"then, the state of others is ",
        "desc":"other's reaction to the event"
    },
    "oReact":{ 
        1:"As a result of {event}, others would feel {ph}.",
        2:"Others feel {ph} because {event}",
        "fa":"در نتیجه دیگران حس می کنند",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"then, the state of others is ",
        "desc":"other's reaction to the event"
    },
    "xReact":{ 
        1:"As a result of {event}, PersonX would feel {ph}. ",
        2:"Others feels {ph} because {event}",
        "fa":"در نتیجه PersonX حس می کند", 
        "tokens":"<state> <agent> <after>",
        "n-tokens":"event {event}, agent state after is {ph}",
        "nat-tokens":"then, the state of the person is ",
        "desc":"the person's reaction to the event"
    },
    "oWant":{ 
        1:"After {event}, others would want {ph}. ",
        2:"Others want {ph} after {event}",
        "fa":"بعد از آن دیگران می خواهند",
        "tokens":"<event> <other> <after> <want>",
        "n-tokens":"event {event}, other after want {ph}",
        "nat-tokens":"then, others want ",
        "desc":"other's decision after the event"
    },
    "xEffect":{ 
        1:"As a result of {event}, PersonX will {ph}. ",
        2:"PersonX {ph} because {event}",
        "fa":"در نتیجه PersonX ",
        "tokens":"<event> <agent> <after> <effect>",
        "n-tokens":"event {event}, agent after effect {ph}",
        "nat-tokens":"then, the effect on the person ",
        "desc":"the effect of event on the person"
    },
    "oEffect":{ 
        1:"as a result of {event}, others will {ph}. ",
        2:"Others {ph} because {event}",
        "fa":"در نتیجه دیگران ",
        "tokens":"<event> <other> <after> <effect>",
        "n-tokens":"event {event}, other after effect {ph}",
        "nat-tokens":"then, the effect on others ",
        "desc":"the effect of the person on others"
    },
    "xWant":{ 
        1:"After {event}, PersonX would want {ph}. ",
        2:"PersonX wants {ph} after {event}",
        "fa":"بعد از آن PersonX می خواهد",
        "tokens":"<event> <agent> <after> <want>",
        "n-tokens":"event {event}, agent after want {ph}",
        "nat-tokens":"then, the person wants ",
        "desc":"the person's decision after the event"
    },
    "xIntent":{ 
        1:"Because of {event}, they want {ph}",
        2:"Because of {event}, they want {ph}",
        3:"Because of {event}, he wants to {ph}",
        4:"Because of {event}, he wants {ph}",
        5:"Before {event}, PersonX would want {ph}. ",
        #2:"PersonX want {ph}  Therefore, {event}",
        "fa":"زیرا PersonX می خواست",
        "tokens":"<event> <agent> <before> <cause> <want>",
        "tokens2":"<event> {event} <agent> <before> <cause> <want> {ph}",
        "random":"event agent before cause  {event} want {ph}",
        "random2":"event agent before cause want {event} {ph}",
        "n-tokens":"event {event}, agent before cause want {ph}",
        "nat-tokens":"before, because the person want ",
        "desc":"the intention of the person"
    },
    "xNeed":{ 
        1:"{event}, Before that, PersonX needs {ph}. ",
        2:"PersonX needs {ph} before {event}",
        "en":"Before that, PersonX needs {ph} ",
        "fa":"قبل از آن PersonX نیاز دارد",
        "tokens":"<event> <agent> <before> <cause> <need>",
        "tokens2":"<event> {event} <agent> <before> <cause> <need> {ph}",
        "random":"event agent before cause  {event} need {ph}",
        "random2":"event agent before cause need {event} {ph}",
        "n-tokens":"event {event}, agent before cause need {ph}",
        "nat-tokens":"before, because the person needs ",
        "desc": "the requirements for the action"
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
              
targets = ["target_text"] #, "target_text_fa", "pred_text1", "all_preds", "pred_text_fa","all_preds_fa", "natural_target_text_fa", "natural_target_text"]
inputs = ["input_text"] #, "input_text_fa", "natural_input_text", "natural_input_text_fa"]

placeholder_token = "<extra_id_0>"
end_token = SPECIAL_TOKENS['eos_token']  #"</s>"
# %%
relation_prompt_lengths = {"com":[3]}

for key, val in rel_nat_maps.items():
    relation_prompt_lengths[key] = [len(val["tokens"].split())] 

def get_prompt_token_fn(id_offset):
    return lambda x: (x>=id_offset) #&(x<id_offset+length)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

encoder_relation_mappings = {}
decoder_relation_mappings = {}

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        #tf.random.set_seed(seed)
        #torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

def tokenize_relations(tokenizer, map_lengths=False):
    for rel,phrase in rel_nat_maps.items():
        natural_rel = phrase[1]
        natural_rel = re.sub(r'{.*?}','', natural_rel)
        #dlog.info("rel ids ***: %s", natural_rel)
        rel_tokens = tokenizer.tokenize(natural_rel)
        rel_nat_maps[rel]["rel_tokens"] = rel_tokens
        #dlog.info("rel ids ***: %s", rel_tokens)
        rel_ids = tokenizer.convert_tokens_to_ids(rel_tokens)
        #dlog.info("rel ids ***: %s", rel_ids)
        rel_nat_maps[rel]["rel_ids"] = rel_ids
        if map_lengths:
            relation_prompt_lengths[rel] = [len(rel_tokens)]

def set_prompt_lengths(rel, length):
    if rel != "":
        relation_prompt_lengths[rel] = length
    else:
        for rel in relation_prompt_lengths.keys():
            relation_prompt_lengths[rel] = length

def extend_tokenizer(tokenizer, prompt_tokens = [], model_id=""):
    cur_list = tokenizer.additional_special_tokens
    num_added_toks: dict = {}
    if tokenizer.bos_token is None:
        num_added_toks['bos_token'] = "<s>"
    if tokenizer.eos_token is None:
        num_added_toks['eos_token'] = "</s>"
    if tokenizer.pad_token is None:
        num_added_toks['pad_token'] = "<pad>"
    if tokenizer.sep_token is None:
        num_added_toks['sep_token'] = "<sep>"
    if tokenizer.cls_token is None:
        num_added_toks['cls_token'] = "<cls>"
    if tokenizer.mask_token is None:
        num_added_toks['mask_token'] = "<mask>"

    num_new_tokens: int = tokenizer.add_special_tokens(num_added_toks)
    rels_tokens = []
    #for x,t in rel_nat_maps.items():
    #    rels_tokens += t["tokens"].split()

    #rels_tokens = list(set(rels_tokens))

    #mlog.info("RELS %s", rels_tokens)
    #new_tokens = tokens.t5_tokens + \
    new_tokens = list(set(rel_maps.values()))+ \
                 list(set(gen_tokens.values())) #+ rels_tokens 
    if prompt_tokens:
        new_tokens += prompt_tokens

    dlog.info(cur_list)
    added_tokens = [ 
            AddedToken(tok,lstrip=True,
                rstrip=False)
            for tok in new_tokens if not tok in cur_list
    ]
    if added_tokens:
        added_tokens = cur_list + added_tokens
        tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
    else:
        mlog.info("No new token was added")


def wrap_model(model, tokenizer, encoder_type="lstm", prompt_path="", flat_prompts=False, method="", shared_embs =False, skilled_variant="", prefix_config=None, exp_id="", encoder_prompts={}, general_prompts={}, n_tasks=2):
    wrapped_model = None
    offsets = []
    tokenize_relations(tokenizer)
    flat_prompt_tokens = []
    ii = 1
    general_encoders = []
    for rel, prompt_tokens in general_prompts.items():
        mlog.info("%s )******************* Wrapping model for %s", ii, rel)
        if rel == "com":
            continue
        for p in prompt_tokens:
            if not p in flat_prompt_tokens:
                flat_prompt_tokens.append(p)
        if not flat_prompts:
            encoder, offset = create_encoder(rel, model, tokenizer, prompt_tokens, encoder_type, wrapped_model, prefix_config)
            general_encoders.append(encoder)
            offsets.append(offset)
            ii += 1
    
    prompt_encoders = []
    for rel, prompt_tokens in encoder_prompts.items():
        mlog.info("%s )******************* Wrapping model for %s", ii, rel)
        if rel == "com":
            continue
        for p in prompt_tokens:
            if not p in flat_prompt_tokens:
                flat_prompt_tokens.append(p)
        if not flat_prompts:
            encoder, offset = create_encoder(rel, model, tokenizer, prompt_tokens, encoder_type, wrapped_model, prefix_config)
            prompt_encoders.append(encoder)
            offsets.append(offset)
            ii += 1
    
    id_offset = len(tokenizer)
    embedding_dim = model.config.hidden_size
    if prompt_encoders:
        id_offset = min(offsets)
#################
        flat_prompt_ids = []
        if prompt_encoders:
           flat_embedding = torch.nn.Embedding(len(flat_prompt_tokens), embedding_dim)
           p_index = 0
           mlog.info("len merge tokens: %s", len(flat_prompt_tokens))
           for encoder in prompt_encoders:
               for pid, emb in encoder.init_embs.items():
                   if not pid in flat_prompt_ids:
                      with torch.no_grad():
                        flat_prompt_ids.append(pid)
                        flat_embedding.weight[p_index] = emb #.detach()
                        p_index += 1
           #_ids_tensor = torch.LongTensor(flat_prompt_ids)
           #embs = model.get_input_embeddings()
           #flat_embs = embs(_ids_tensor)
           #with torch.no_grad():
           #    for i, e in enumerate(flat_embs):
           #        flat_embedding.weight[i] = e #.detach()
           if shared_embs:
               mlog.info("Flat_ids: %s", flat_prompt_ids)
               for encoder in prompt_encoders:
                    encoder.embedding = flat_embedding
                    #encoder.id_offset= -1
                    #encoder.length= len(flat_prompt_tokens)

    flat_encoder = None 
    merge_encoder = None
    flat_embedding = None
    n_prompt_tokens = len(flat_prompt_tokens)
    mbp("")
    for encoder in prompt_encoders:
        if isinstance(encoder, MergePromptEncoder):
            encoder.encoders = general_encoders
            encoder.trunc_router = main_args["trunc_router"]

    if flat_prompts:
        flat_encoder, _ = create_encoder("flat", model, tokenizer, flat_prompt_tokens, flat_prompts, wrapped_model, prefix_config)
        #assert flat_encoder != None, "merge encoder for " + flat_prompts + " is none"
        #flat_encoder = _encoder # prompt_encoders[0]
        #prompt_encoders = [_encoder]
####################
    mlog.info("ID OFFSET: %s", id_offset)
    if skilled_variant:
       wrapped_model = SkilledMixin(model, n_tasks=n_tasks, n_skills=len(general_encoders), 
               skilled_variant=skilled_variant,
               prompt_encoders=prompt_encoders, 
               general_encoders=general_encoders, 
               prompt_token_fn=get_prompt_token_fn(id_offset), 
               merge_encoder=merge_encoder, flat_encoder=flat_encoder) 
    else:
        wrapped_model = PTuningWrapper(model, 
                prompt_encoders, 
                general_encoders=general_encoders, 
                prompt_token_fn=get_prompt_token_fn(id_offset), 
                merge_encoder=merge_encoder, flat_encoder=flat_encoder, 
                exp_id=exp_id)
    return wrapped_model

def create_encoder(name, model, tokenizer, prompt_tokens, encoder_type="lstm", 
        wrapped_model = None, prefix_config=None):
    embedding_dim = model.config.hidden_size
    enc_plen = len(prompt_tokens)

    rel_tokens = prompt_tokens #+ common_tokens
    mlog.info("** rel tokens : %s", rel_tokens)
    cur_list = tokenizer.additional_special_tokens
    my_specials = [x for x in cur_list if not "<extra_id"  in x]
    #mlog.info("** cur tokens : %s", my_specials)

    enc_plen =len(rel_tokens) 
    mlog.info("** len tokenizer before extend: %s", len(tokenizer))
    extend_tokenizer(tokenizer, rel_tokens)
    model.resize_token_embeddings(len(tokenizer))
    rel_ids = tokenizer.convert_tokens_to_ids(rel_tokens)
    cur_embeddings = model.get_input_embeddings()
    init_embs = {}
    if "@" in name:
        name, encoder_type = name.split("@") 
    for p in rel_tokens:
        if "_" in p:
           p = p.strip("<").strip(">")
           w = p.split("_")[1]
           if not w.isdigit():
               wid = tokenizer.convert_tokens_to_ids([w])[0]
               emb = cur_embeddings.weight[wid,:].detach().clone() 
               pid = tokenizer.convert_tokens_to_ids([p])[0]
               init_embs[pid] = emb
        #else:
        #   pid = tokenizer.convert_tokens_to_ids([p])[0]
        #   emb = cur_embeddings.weight[pid,:].detach().clone() 
        #   init_embs[pid] = emb
    mlog.info("** final rel ids: %s", rel_ids)
    id_offset = min(rel_ids) 
    prompt_encoder = None
    mlog.info("Encoder Type %s", encoder_type)
    mlog.info("id_offset: %s", id_offset)
    mlog.info("enc_plan: %s", enc_plen)
    mlog.info("enc prompts: %s", rel_tokens)

    if encoder_type.startswith("mlp"):
        mlog.info("in Emb %s", encoder_type)
        _enc_type = encoder_type.split("@")
        num_layers = 1
        if len(_enc_type) > 1:
            num_layers = int(_enc_type[1])
        hidden_size = -1
        if len(_enc_type) > 2:
            hidden_size = int(_enc_type[2])
        if enc_plen > 0:
            mlog.info("Prompt Encoder defined : %s", enc_plen)
            prompt_encoder = MLPPromptEncoder(name, enc_plen,
                    embedding_dim,id_offset = -1, prompt_ids=rel_ids, init_embs=init_embs, num_layers=num_layers, hidden_size=hidden_size)
    elif encoder_type.startswith("emb"):
        mlog.info("in Emb %s", encoder_type)
        if enc_plen > 0:
            mlog.info("Prompt Encoder defined : %s", enc_plen)
            prompt_encoder = EmbeddingPromptEncoder(name, enc_plen,
                    embedding_dim,id_offset = -1, prompt_ids=rel_ids, init_embs=init_embs)
    elif encoder_type.startswith("merge"):
        mlog.info("in Merge %s", encoder_type)
        if enc_plen > 0:
            mlog.info("Prompt Encoder defined : %s", enc_plen)
            prompt_encoder = MergePromptEncoder(name = name, length=enc_plen,
                    embedding_dim = embedding_dim,
                    id_offset = -1, prompt_ids=rel_ids, init_embs=init_embs)
    elif encoder_type.startswith("mat"):
        mlog.info("in Mat %s", encoder_type)
        if enc_plen > 0:
            mlog.info("Prompt Encoder defined : %s", enc_plen)
            prompt_encoder = MatPromptEncoder(prefix_config=prefix_config, 
                    name = name, length=enc_plen,
                    embedding_dim = embedding_dim,
                    id_offset = -1, prompt_ids=rel_ids, init_embs=init_embs)
    else:
        if enc_plen > 0:
            _enc_type = encoder_type.split("@")
            num_layers = 1
            hidden_size = -1
            if len(_enc_type) > 1:
                num_layers = int(_enc_type[1])
            if len(_enc_type) > 2:
                hidden_size = int(_enc_type[2])
            mlog.info("Prompt Encoder defined : %s", enc_plen)
            prompt_encoder = LSTMEmbeddingPromptEncoder(name, enc_plen,embedding_dim,
                    id_offset = -1, prompt_ids=rel_ids, init_embs=init_embs, num_layers=num_layers, hidden_size=hidden_size)


    return prompt_encoder, id_offset

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

def save_checkpoint(model, tokenizer, optimizer, scheduler, step, 
                   best_eval_step, best_dev_loss, save_path):
    if "/temp/" in save_path:
        mlog.info("Saves in temp are skipped ")
        return

    mlog.info("Saving model ... %s", save_path)
    with open(save_path + "/best_model.txt", "a") as f:
        print("best_step:", best_eval_step, file=f)
        print("best dev loss:", best_dev_loss, file=f)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

#    torch.save(model.state_dict(), os.path.join(save_path,"state_dict"))
#    torch.save(model, os.path.join(save_path,"mymodel"))
#    torch.save({
#            'step': step,
#            'eval_step': best_eval_step,
#            'model_state_dict': model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'scheduler_state_dict': scheduler.state_dict(),
#            }, os.path.join(save_path, "saved_states"))


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
def save_data(ex_df, save_ds_path):
    if save_ds_path and len(ex_df) > 0:
        ex_df = ex_df.drop_duplicates(["input_text","prefix"])
        ex_df = ex_df.sort_values(by=["input_text","prefix"])
        mlog.info("DF saved as %s", save_ds_path)
        ex_df.to_csv(save_ds_path, index=False, sep="\t")

def fix_pos(qtemp, gen_pos, prompt_pos):
   if gen_pos == "end":
       qtemp = qtemp.replace("{gen_start} ","")
       qtemp = qtemp.replace("{gen_end}","{gen}")
   else:
       qtemp = qtemp.replace(" {gen_end}","")
       qtemp = qtemp.replace("{gen_start}","{gen}")
   if prompt_pos == "end":
       qtemp = qtemp.replace("{rel_i_start} ","")
       qtemp = qtemp.replace("{rel_i_end}","{rel_i}")
   else:
       qtemp = qtemp.replace(" {rel_i_end}","")
       qtemp = qtemp.replace("{rel_i_start}","{rel_i}")
   return qtemp
def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
