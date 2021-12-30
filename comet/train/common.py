from pathlib import Path
import math
from termcolor import colored
from transformers import AddedToken 
import pandas as pd
from comet.utils.myutils import *
from comet.transformers_ptuning import PTuningWrapper
from comet.transformers_ptuning.ptuning_wrapper import LSTMEmbeddingPromptEncoder, EmbeddingPromptEncoder
from tqdm import tqdm
import logging, sys
import random
import tokens
import re
import os
import torch
import json
from mylogs import *
import pickle5 as pickle

SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "</s>",
                    "unk_token": "<|UNK|>",
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}

sep = "<|SEP|>"
pad_token = {"pad_token": "<|PAD|>"}
sep_token = {"sep_token": sep}
nli_map = ['contradiction', 'entailment', 'neutral']
all_rels = ["oEffect","oReact", "oWant", "xAttr", "xEffect","xIntent","xNeed","xReact","xWant"]
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
        "en":"As a result others feel ",
        "fa":"در نتیجه دیگران حس می کنند",
        "tokens":"<state> <other> <after>",
        "nat-tokens":"then, the state of others is "
    },
    "xReact":{ 
        "en":"As a result PersonX feels ",
        "fa":"در نتیجه PersonX حس می کند", 
        "tokens":"<state> <agent> <after>",
        "nat-tokens":"then, the state of the person is "
    },
    "xWant":{ 
        "en":"Then PersonX wants ",
        "fa":"بعد از آن PersonX می خواهد",
        "tokens":"<event> <agent> <after> <want>",
        "nat-tokens":"then, the person wants "
    },
    "oWant":{ 
        "en":"Then others want ",
        "fa":"بعد از آن دیگران می خواهند",
        "tokens":"<event> <other> <after> <want>",
        "nat-tokens":"then, others want "
    },
    "xEffect":{ 
        "en":"As a result PersonX  ",
        "fa":"در نتیجه PersonX ",
        "tokens":"<event> <agent> <after> <effect>",
        "nat-tokens":"then, the effect on the person "
    },
    "oEffect":{ 
        "en":"As a result others  ",
        "fa":"در نتیجه دیگران ",
        "tokens":"<event> <other> <after> <effect>",
        "nat-tokens":"then, the effect on others "
    },
    "xAttr":{ 
        "en":"PersonX is seen as",
        "fa":"مردم فکر می کنند PersonX ",
        "tokens":"<state> <agent> <static>",
        "nat-tokens":"always, the state of the person is",
    },
    "xIntent":{ 
        "en":"Because PersonX intended ",
        "fa":"زیرا PersonX می خواست",
        "tokens":"<event> <agent> <before> <cause> <want>",
        "nat-tokens":"before, because the person want "
    },
    "xNeed":{ 
        "en":"Before that, PersonX needs ",
        "fa":"قبل از آن PersonX نیاز دارد",
        "tokens":"<event> <agent> <before> <cause> <need>",
        "nat-tokens":"before, because the person needs "
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
relation_prompt_lengths = {
    "xAttr":[4],
    "xEffect":[8],
    "oEffect":[8],
    "xReact":[5],
    "oReact":[5],
    "xWant":[5],
    "oWant":[5],
    "xIntent":[5],
    "xNeed":[5],
    "com":[3],
}


def get_prompt_token_fn(id_offset):
    return lambda x: (x>=id_offset) #&(x<id_offset+length)

encoder_relation_mappings = {}
decoder_relation_mappings = {}

def tokenize_relations(tokenizer, map_lengths=False):
    for rel,phrase in relation_natural_mappings.items():
        natural_rel = phrase["en"]
        #dlog.info("rel ids ***: %s", natural_rel)
        rel_tokens = tokenizer.tokenize(natural_rel)
        relation_natural_mappings[rel]["rel_tokens"] = rel_tokens
        #dlog.info("rel ids ***: %s", rel_tokens)
        rel_ids = tokenizer.convert_tokens_to_ids(rel_tokens)
        #dlog.info("rel ids ***: %s", rel_ids)
        relation_natural_mappings[rel]["rel_ids"] = rel_ids
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
    rels_tokens = []
    for x,t in relation_natural_mappings.items():
        rels_tokens += t["tokens"].split()

    rels_tokens = list(set(rels_tokens))

    mlog.info("RELS %s", rels_tokens)
    new_tokens = tokens.t5_tokens + \
                 list(atomic_relation_mappings.values())+ \
                 list(gen_tokens.values()) + rels_tokens
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

def fill_sample(mt, rel):
    qtemp, anstemp, ex_qtemp, ex_anstemp, context = create_templates(mt, 
            gen_pos="end")
    mask =1
    context_df = None
    d = {"prefix":rel}
    event = "test event"
    resp = "test answer"
    input_lang = "en"
    target_lang = "en"
    gen_token = "gen_en"
    _qtemp = fill_consts(qtemp, ex_qtemp, context,d, context_df, mask=mask,method = mt)
    _anstemp = fill_consts(anstemp, ex_anstemp, context,d, context_df, mask=mask,method = mt)
    _query = fill_vars(_qtemp, rel, event, gen_token, resp, 
            input_lang, target_lang) 
    response = fill_vars(_anstemp, rel, event, gen_token, resp, 
            input_lang, target_lang)

def wrap_model(model, tokenizer, encoder_type="lstm", prompt_path="", from_words=False, merge_prompts=False, method=""):
    wrapped_model = None
    prompt_encoders = []
    offsets = []
    tokenize_relations(tokenizer)
    for rel in all_rels:
        fill_sample(method, rel)

    for rel, prompt_tokens in encoder_prompts.items():
        mlog.info("******************* Wrapping model for %s", rel)
        if rel == "com":
            continue
        if from_words == "rel":
            from_words = relation_natural_mappings[rel]["en"]
        if from_words == "rel_tokens":
            prompt_tokens = relation_natural_mappings[rel]["rel_tokens"]

        encoder, offset = create_encoder(rel, model, tokenizer, prompt_tokens, encoder_type, from_words, wrapped_model)
        prompt_encoders.append(encoder)
        offsets.append(offset)
    id_offset = min(offsets)
    mlog.info("ID OFFSET: %s", id_offset)
    wrapped_model = PTuningWrapper(model, prompt_encoders, prompt_token_fn=get_prompt_token_fn(id_offset), merge_prompts=merge_prompts)
    return wrapped_model

def create_encoder(name, model, tokenizer, prompt_tokens, encoder_type="lstm", 
        from_words=False, wrapped_model = None):
    embedding_dim = model.config.hidden_size
    enc_plen = len(prompt_tokens)

    rel_tokens = prompt_tokens + common_tokens
    mlog.info("** rel tokens : %s", rel_tokens)
    cur_list = tokenizer.additional_special_tokens
    my_specials = [x for x in cur_list if not "<extra_id"  in x]
    mlog.info("** cur tokens : %s", my_specials)


    enc_plen =len(rel_tokens) 
    mlog.info("** len tokenizer before extend: %s", len(tokenizer))
    extend_tokenizer(tokenizer, rel_tokens)
    rel_ids = tokenizer.convert_tokens_to_ids(rel_tokens)
    mlog.info("** final rel ids: %s", rel_ids)
    id_offset = min(rel_ids) 
    prompt_encoder = None
    mlog.info("Encoder Type %s", encoder_type)
    mlog.info("id_offset: %s", id_offset)
    mlog.info("enc_plan: %s", enc_plen)
    mlog.info("enc prompts: %s", rel_tokens)
    if encoder_type.startswith("emb"):
        mlog.info("in Emb %s", encoder_type)
        if enc_plen > 0:
            mlog.info("Prompt Encoder defined : %s", enc_plen)
            prompt_encoder = EmbeddingPromptEncoder(name, enc_plen,
                    embedding_dim,id_offset = -1, prompt_ids=rel_ids)
    else:
        if enc_plen > 0:
            mlog.info("Prompt Encoder defined : %s", enc_plen)
            prompt_encoder = LSTMEmbeddingPromptEncoder(name, enc_plen,embedding_dim,
                    id_offset = -1, prompt_ids=rel_ids)

    model.resize_token_embeddings(len(tokenizer))

    return prompt_encoder, id_offset

encoder_prompts = {} 
decoder_prompts = {}
def fill_const_for_rel(template, row):
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
            "{sep}":sep,
            "{gen_en}":gen_token_en,
            "{end}":end_token}
    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], template)
    for key,value in row.items():
        val = str(value)
        text = text.replace("{" + key + "}", val)
    return text
common_tokens = []
def fill_prompt(text, rel, place_holder, counter = 0, lang=""):
    pi = 0
    plen = relation_prompt_lengths[rel]
    _pholder = place_holder
    place_holder = place_holder.replace("{", "<")  
    place_holder = place_holder.replace("}", ">")  
    place_holder = place_holder.replace("rel", rel)  
    place_holder = place_holder.replace("lang", lang)  
    #dlog.info("text: %s", text)
    while _pholder in text:
        if "_i" in _pholder:
            enc_plen = plen[pi] if pi < len(plen) else plen[-1] 
            prompt = ""
            for i in range(counter, counter + enc_plen):
                token = place_holder
                token = token.replace("_i", "_" + str(i))  
                prompt += " " + token
        elif _pholder == "{tokens}": 
            prompt = relation_natural_mappings[rel]["tokens"]
        else:
            mlog.info("************** using tokens of pholder %s",_pholder)
            prompt = place_holder
        prompt = prompt.strip()
        enc_plen = len(prompt.split())
        for token in prompt.split():
            if rel == "com" and not token in common_tokens:
                common_tokens.append(token)
            else:
                if not rel in encoder_prompts:
                    encoder_prompts[rel] = []
                if not token in encoder_prompts[rel]:
                    encoder_prompts[rel].append(token)
        text = text.replace(_pholder,prompt, 1)
        counter += enc_plen 
        pi += 1
    #dlog.info("text: %s", text)
    return text

def fill_consts(template, ex_temp, context, row, rows=None, mask=-1, method=""):
    #dlog.info("fill consts, input text: %s", template)
    text = fill_const_for_rel(template, row)
    #dlog.info("fill consts, input text: %s", text)
    rel = row["prefix"]
    plen = relation_prompt_lengths[rel]
    text = text.replace("{ph}", placeholder_token)
    #dlog.info("fill consts, input text: %s", context)
    if not rel in encoder_prompts:
        encoder_prompts[rel] = []
    if not rel in decoder_prompts:
        decoder_prompts[rel] = []
    #sent = "This is a good apple".split(" ")
    if mask >= 0 and "{enc_token_mask}" in text:
        prompt = f"<enc_mask_{mask}>" 
        text = text.replace("{enc_token_mask}",prompt)
    while "{rel_fw}" in text:
        rel_ids = relation_natural_mappings[rel]["rel_ids"]
        prompt = ""
        for i in range(len(rel_ids)):
            token = f"<{rel}_{i}>" 
            if not token in encoder_prompts[rel]:
                encoder_prompts[rel].append(token)
            prompt += " " + token
        prompt = prompt.strip()
        text = text.replace("{rel_fw}",prompt, 1)
    counter = 0
    pi = 0
    enc_prompt = ""
    dec_prompt = ""
    while "{enc_token_rest}" in text:
        enc_plen = plen[pi] if pi < len(plen) else plen[-1] 
        prompt = ""
        for i in range(counter, counter + enc_plen):
            token = f"<enc_mask_{i}>" 
            if not token in encoder_prompts[rel]:
                encoder_prompts[rel].append(token)
            if i == mask:
               token = "<extra_id_0>"
            prompt += " " + token
        prompt = prompt.strip()
        if not enc_prompt:
            enc_prompt = prompt
        text = text.replace("{enc_token_rest}",prompt, 1)
        counter += enc_plen 
        pi += 1
    counter = mask
    pi = 0
    enc_prompt = ""
    dec_prompt = ""
    while "{enc_token_cont}" in text:
        enc_plen = plen[pi] if pi < len(plen) else plen[-1] 
        prompt = ""
        for i in range(counter, enc_plen):
            token = f"<enc_mask_{i}>" 
            if not token in encoder_prompts[rel]:
                encoder_prompts[rel].append(token)
            prompt += " " + token
        prompt = prompt.strip()
        if not enc_prompt:
            enc_prompt = prompt
        text = text.replace("{enc_token_cont}",prompt, 1)
        counter += enc_plen 
        pi += 1
    #dlog.info("encoder prompt %s ", encoder_prompts[rel])
    counter = 0
    pi = 0
    enc_prompt = ""
    dec_prompt = ""
    while "{enc_com_token}" in text:
        enc_plen = plen[pi] if pi < len(plen) else plen[-1] 
        prompt = ""
        for i in range(counter, counter + enc_plen):
            token = f"<enc_com_{i}>" 
            prompt += " " + token
            if not token in encoder_prompts[rel]:
                encoder_prompts[rel].append(token)
        prompt = prompt.strip()
        if not enc_prompt:
            enc_prompt = prompt
        text = text.replace("{enc_com_token}",prompt, 1)
        counter += enc_plen 
        pi += 1
    text = fill_prompt(text, rel, "{rel_i}")
    text = fill_prompt(text, "com", "{com_i}")
    text = fill_prompt(text, rel, "{tokens}")
    if "{examples}" in text:
        examples = ""
        assert rows is not None and len(rows) > 0, "Since there are examples in template, rows must be provided"
        ii = 1
        for idx, _row in rows.iterrows():
            example = ex_temp
            numbered = False
            if "{num}" in ex_temp:
                numbered = True
                example = example.replace("{num}", "")
            #dlog.info("example: %s", _row)
            if "{rel_i}" in ex_temp:
                assert enc_prompt != "", "Prompt was not set!"
            example = fill_const_for_rel(example, _row)
            example = fill_prompt(example, rel, "{rel_i}")
            example = fill_prompt(example, rel, "{tokens}")
            for key,value in _row.items():
                val = str(value)
                if "fa" in method and "_fa" in key:
                    val = toPers(val)
                example = example.replace("{" + key + "}", val)
            if numbered: 
                examples += " " + str(ii) + ") " + example
            else:
                examples += example
            ii += 1

        text = text.replace("{examples}", examples + " " + str(ii) + ")")
    ## storyyyy
    ii = 1
    if rows is not None:
        for idx, _row in rows.iterrows():
            example = ex_temp
            relation = _row["prefix"]
            if relation == rel:
                continue
            numbered = False
            if "{num}" in ex_temp:
                numbered = True
                example = example.replace("{num}", "")
            if "{rel_i}" in ex_temp:
                assert enc_prompt != "", "Prompt was not set!"
            example = fill_const_for_rel(example, _row)
            example = fill_prompt(example, relation, "{rel_i}")
            example = fill_prompt(example, "com", "{com_i}")
            for key,value in _row.items():
                val = str(value)
                if "fa" in method and "_fa" in key:
                    val = toPers(val)
                example = example.replace("{" + key + "}", val)
            if numbered: 
                example = " " + str(ii) + ") " + example
            context = context.replace("{" + relation + "}", example)
            ii += 1
    for relation in all_rels:
        if relation == rel:
            context = context.replace("{" + relation + "}", text)
        else:
            context = context.replace("{" + relation + "}", "")
    ii = 1
    while "{ph}" in context:
        ph = f"<extra_id_{ii}>"
        ii += 1
        context = context.replace("{ph}", ph, 1)

    #dlog.debug("after: %s", context)
    return context 

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
def create_templates(method, gen_pos="end", prompt_pos="end"):
       ex_qtemp = ""
       ex_anstemp = ""
       context = "{xIntent} {xAttr} {xNeed} {xReact} {oReact} {xWant} {oWant} {xEffect} {oEffect}"
       if method == "bart":
           qtemp = "{event} {rel} [GEN]"
           anstemp = "{resp}"
       elif method == "blank":
           qtemp = "{event} {rel_natural} {resp}"
           anstemp = "blank"
       elif method == "pred-emb":
           qtemp = "{enc_token_rest}"
           anstemp = "{ph} {enc_token_mask}"
       elif method == "pred-emb-rev":
           qtemp = "{enc_token_mask} {ph}"
           anstemp = "{ph} {enc_token_cont}"
       elif method == "rel-enc":
           qtemp = "{event} {rel_i} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "rel-dec":
           qtemp = "{event} {ph} {resp}"
           anstemp = "{ph} {rel_i}"
       elif method == "rel-mask":
           qtemp = "{event} {ph} {resp}"
           anstemp = "{ph} {rel_natural}"
       elif method == "unsup-rel":
           qtemp = "{event} {ph} {resp}"
           anstemp = "{ph} {prefix}"
       elif method == "unsup-wrap-rel":
           qtemp = "{com_i} {event} {ph} {resp}"
           anstemp = "{prefix}"
       elif method == "rel-mask-wrap":
           qtemp = "{enc_com_token} {event} {ph} {resp}"
           anstemp = "{ph} {rel_natural}"
       elif method == "rel-unsup":
           qtemp = "{event} {rel_natural} {ph} "
           anstemp = "{ph} {resp}"
       elif method == "sup-pred-enfa":
           qtemp = "{input_text} {rel_i} {gen_fa}"
           anstemp = "{input_text_fa} {dec_token} {target_text_fa}"
       elif method == "sup-enfa":
           qtemp = "{input_text} {rel_i} {target_text} {gen_fa}"
           anstemp = "{input_text_fa} {dec_token} {target_text_fa}"
       elif method == "sup-enmix":
           qtemp = "{input_text} {rel_i} {target_text} {gen}"
           anstemp = "{event} {dec_token} {gen} {resp}"
       elif method == "unsup-nat-gen":
           qtemp = "{rel_i} {event} {rel_natural} {gen} {ph}" 
           anstemp = "{ph} {resp} {end}"
       elif method == "sup-nat-tokens":
           qtemp = "{event} {nat_tokens}" 
           anstemp = "{resp} {end}"
       elif method == "unsup-nat-tokens":
           qtemp = "{event} {nat_tokens} {ph}" 
           anstemp = "{ph} {resp} {end}"
       elif method == "sup-nat":
           qtemp = "{event} {rel_natural}" 
           anstemp = "{resp} {end}"
       elif method == "unsup-nat":
           qtemp = "{event} {rel_natural} {ph}" 
           anstemp = "{ph} {resp} {end}"
       elif method == "enc-unsup-nat":
           qtemp = "{rel_i} {event} {rel_natural} {ph}" 
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup-nat-fa":
           qtemp = "{event} {rel_natural_fa} {ph}" 
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup-wrap-nat":
           qtemp = "{rel_i} {event} {rel_natural} {ph}" 
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup-wrap-lang":
           qtemp = "{rel_i} {event} {rel_lang_i} {ph}" 
           anstemp = "{ph} {resp} {end}"
       elif method == "sup-gen":
           qtemp = "{event} {gen}"
           anstemp = "{resp} {end}"
       elif method == "sup-wrap":
           qtemp = "{rel_i_start} {event} {rel_i_end} {gen}"
           anstemp = "{resp} {end}"
       elif method == "sup-no-gen":
           qtemp = "{event}"
           anstemp = "{resp} {end}"
       elif method == "gen":
           qtemp = "{gen}"
           anstemp = "{resp}"
       elif method == "pred-enfa":
           qtemp = "{rel_i_start} {gen_start} {input_text} {rel_natural_en} {gen_en} {ph} {event} {rel_natural} {rel_i_end} {gen_end} <extra_id_1>"
           anstemp = "{ph} {target_text} <extra_id_1> {resp} <extra_id_2>"
       elif method == "context-en":
           qtemp = "{rel_i_start} {gen_start} {input_text} {rel_natural_en} {gen_en} {target_text} {rel_i_start} {event} {rel_natural} {rel_i_end} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "context-faen":
           qtemp = "{rel_i_start} {gen_start} {input_text_fa} {rel_natural_fa} {gen_en} {target_text} {event} {rel_natural} {rel_i_end} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup-wrap-n-example":
           qtemp = "{examples} {event} {rel_i} {gen} {ph}"
           ex_qtemp = "{input_text} {rel_i} {target_text} {end}"
           anstemp = "{ph} {resp} {end}"
       elif method == "trans":
           qtemp = "{target_text} en2fa"
           anstemp = "{target_text_fa}"
       elif method == "unsup-n-example":
           qtemp = "{examples} {gen} {ph}"
           ex_qtemp = "{gen} {input_text} {end} \n"
           anstemp = "{ph} {event} {end}"
       elif method == "story-wrap":
           qtemp = "{rel_i} {event} {rel_natural} {ph}"
           context = "{xAttr} {xIntent} {xReact}"
           ex_qtemp = "{rel_enc_token} {rel_natural_en} {target_text} {end}"
           anstemp = "{ph} {resp} {end}"
       elif method == "event-resp-n-wrap":
           qtemp = "{event} {examples} {rel_i} {event} {rel_natural} {ph}"
           ex_qtemp = "{rel_natural_en} {target_text} {end} \n"
           anstemp = "{ph} {resp} {end}"
       elif method == "gpt-event-resp-n-wrap":
           qtemp = "{examples} {rel_i}"
           ex_qtemp = "{rel_i} {input_text} {rel_natural_en} {target_text} {sep} \n"
           anstemp = "{event} {rel_natural} {resp} {end}"
       elif method == "fa-gpt-event-resp-n-wrap":
           qtemp = "{examples} {rel_i}"
           ex_qtemp = "{rel_i} {input_text_fa} {rel_natural_fa} {target_text_fa} {sep} \n"
           anstemp = "{event} {rel_natural} {resp} {end}"
       elif method == "unsup-wrap-n-example-fa":
           qtemp = "{examples} {gen} {ph}"
           ex_qtemp = "{gen} {input_text_fa} {end} \n"
           anstemp = "{ph} {event} {end}"
       elif method == "gpt-event-n":
           qtemp = "{examples} {gen}"
           ex_qtemp = "{gen} {input_text} {end} \n"
           anstemp = "{event} {end}"
       elif method == "gpt-fa-event-n":
           qtemp = "{examples} {gen}"
           ex_qtemp = "{gen} {input_text_fa} {end} \n"
           anstemp = "{event} {end}"
       elif method == "unsup-wrap-n-example":
           qtemp = "{examples} {rel_i} {ph}"
           ex_qtemp = "{rel_i} {input_text} {end} \n"
           anstemp = "{ph} {event} {end}"
       elif method == "gpt-n-example":
           qtemp = "{examples} {event} {rel_natural}"
           ex_qtemp = "{input_text} {rel_natural} {target_text} {end}"
           anstemp = "{resp} {end}"
       elif method == "gpt-wrap-n-example":
           qtemp = "{examples} {event} {rel_i}"
           ex_qtemp = "{input_text} {rel_i} {target_text} {end}"
           anstemp = "{resp} {end}"
       elif method == "unsup-wrap-context-n-dec":
           qtemp = "{event} {rel_i} {gen} {ph}"
           ex_qtemp = "{input_text} {target_text} {end}"
           anstemp = "{examples} {ph} {resp} {end}"
       elif method == "unsup-wrap-context-enfa":
           qtemp = "{rel_i_start} {gen_start} {input_text} {rel_natural_en} {gen_fa} {target_text_fa} {rel_i_start} {event} {rel_natural} {rel_i_end} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup-wrap-context-fa":
           qtemp = "{rel_i_start} {gen_start} {input_text_fa} {rel_natural_fa} {gen_fa} {target_text_fa} {rel_i_start} {event} {rel_natural} {rel_i_end} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "sup":
           qtemp = "{rel_token} {event}"
           anstemp = "{resp} {end}"
       elif method == "sup-wrap-gen":
           qtemp = "{rel_i_start} {gen_start} {event} {rel_i_end} {gen_end}"
           anstemp = "{resp} {end}"
       elif method == "gpt-wrap-tokens":
           qtemp = "{examples} {tokens} {event} "
           ex_qtemp = "{tokens} {input_text} {target_text} {end}"
           anstemp = "{resp} {end}"
       elif method == "gpt-wrap":
           qtemp = "{event} {rel_i}"
           anstemp = "{resp} {end}"
       elif method == "gpt":
           qtemp = "{event} {rel_natural}"
           anstemp = "{resp} {end}"
       elif method == "unsup-wrap-fw":
           qtemp = "{event} {rel_fw} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup":
           qtemp = "{rel_token} {event} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup-gen":
           qtemp = "{rel_token} {event} {gen} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup-wrap-com":
           qtemp = "{com_i} {event} {rel_i} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup-wrap":
           qtemp = "{rel_i_start} {event} {rel_i_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "sup-tokens" or "sup-tokens-wrap":
           qtemp = "{event} {tokens}"
           anstemp = "{resp} {end}"
       elif method == "unsup-tokens" or "unsup-tokens-wrap":
           qtemp = "{event} {tokens} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "sup-tokens-start":
           qtemp = "{tokens} {event}"
           anstemp = "{resp} {end}"
       elif method == "unsup-tokens-start":
           qtemp = "{tokens} {event} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup-tokens-gen":
           qtemp = "{event} {tokens} {gen_lang} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup-wrap-gen":
           qtemp = "{rel_i_start} {gen_start} {event} {rel_i_end} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup-wrap-dec":
           qtemp = "{rel_i_start} {gen_start} {event} {rel_i_end} {gen_end} {ph}"
           anstemp = "{ph} {dec_token} {resp} {end}"
       elif method == "unsup-wrap-2":
           qtemp = "{rel_i} {gen_start} {event} {rel_i} {gen_end} {ph}"
           anstemp = "{ph} {resp} {end}"
       elif method == "unsup-wrap-3":
           qtemp = "{rel_i} {gen_start} {event} {rel_i} {gen_end} {ph}"
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
           qtemp = qtemp.replace("{rel_i_start} ","")
           qtemp = qtemp.replace("{rel_i_end}","{rel_i}")
       else:
           qtemp = qtemp.replace(" {rel_i_end}","")
           qtemp = qtemp.replace("{rel_i_start}","{rel_i}")
       while "  " in qtemp:
           qtemp = qtemp.replace("  "," ")


       return qtemp, anstemp, ex_qtemp, ex_anstemp, context

def fill_vars(template, rel, event, gen_token, resp, inp_lang, resp_lang):
    rel_natural = relation_natural_mappings[rel][inp_lang]        
    rel_natural_tokens = relation_natural_mappings[rel]["nat-tokens"]        
    rep  = {"{event}":event, 
            "{resp}":resp,
            "{rel_natural}":rel_natural,
            "{nat_toekns}":rel_natural_tokens,
            "{gen}":gen_token}
    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], template)
    lang = resp_lang
    plen = relation_prompt_lengths[rel]
    if not rel in encoder_prompts:
        encoder_prompts[rel] = []
    if not rel in decoder_prompts:
        decoder_prompts[rel] = []
    text = fill_prompt(text, rel, "{rel_lang_i}", lang=lang)
    text = fill_prompt(text, rel, "{gen_lang}", lang=lang)
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

class MyDataset(torch.utils.data.IterableDataset):
    def __init__(self, split_df, split_name, method, prompt_pos, rel_filter, 
            num_samples=0, 
            ignore_blanks=False,
            only_blanks=False,
            inp_include="",
            inp_exclude="",
            targ_include="",
            targ_exclude="",
            pred_tresh=0,
            nli_group="all", per_record=False, is_even=False, start=0, 
            sampling=0, ex_type="",  samples_per_head=0, save_ds_path=""): 
        super(MyDataset).__init__()
        self.flat_data = []
        self.data_split = {}

        self.only_blanks = only_blanks
        self.samples_per_head = samples_per_head
        self.start = start
        self.prompt_pos = prompt_pos
        self.inp_include = inp_include
        self.inp_exclude = inp_exclude
        self.targ_include = targ_include
        self.targ_exclude = targ_exclude
        self.ex_type = ex_type
        self.per_record = per_record
        self.is_even = is_even
        dlog.info("building query responses for {}".format(split_name))
        mlog.info(f"fill data input dataset len:{len(split_df)}")
        self.natural = inp_include == "natural"
        self.split_name = split_name
        if split_name != "train":
            self.start = 0
        if self.natural and split_name != "train": 
            self.natural = False 
        if self.natural:
            dlog.info("natural is ON")
        self.num_samples = num_samples
        if self.num_samples == 0: 
            self.num_samples = len(split_df)
            self.samples_per_head = 0
            self.is_even = False
        for col in targets:
            if col in split_df:
                split_df[col] = split_df[col].astype(str)
        if ignore_blanks: # and len(split_df) > num_rows:
            split_df = split_df[split_df["input_text"].str.contains('___')==False]
            #split_df = split_df[split_df["target_text"] != "none"]
            dlog.info("*** Filtered for ignoring blanks ")
        elif only_blanks:
            split_df = split_df[split_df["input_text"].str.contains('___')==True]
            #split_df = split_df[split_df["target_text"] != "none"]
            dlog.info("*** Filtered for including only with blanks ")
        if pred_tresh > 0 and "bert_score" in split_df:
            split_df = split_df[split_df["bert_score"] > pred_tresh]
            dlog.info("*** Filtered based on pred1 score higher than "+ pred_tresh)
        if nli_group != "all" and "nli_group" in split_df:
            split_df = split_df[split_df["nli_group"] == nli_group]
            dlog.info("*** Filtered based on nli_group "+ nli_group)

        mlog.info(f"len after filtering:{len(split_df)}")
        assert len(split_df) > 0, "Data frame is empty " + self.split_name
        self.num_records = self.num_samples
        if rel_filter:
            split_df = split_df[split_df["prefix"] == rel_filter]
            dlog.info("len after relation filter: %s", len(split_df))
        if False:
            if self.num_samples < len(split_df) and not is_even: 
                #TODO 
                split_df = split_df.groupby("prefix").sample(n=self.num_samples)
                self.num_records = len(split_df)
                dlog.info(f"NUM samples %s, %s", self.num_samples, len(split_df))
                dlog.info(f"len after sampling:{len(split_df)}")
        self.split_df = split_df.sort_values(by="input_text")
        assert len(self.split_df) > 0, "Data frame is empty " + self.split_name + " " + str(self.num_samples)
        self.cats_num = cats_num = len(split_df["prefix"].unique())
        dlog.info("Num Samples: %s", self.num_samples)
        mlog.info("Cats Num: %s", cats_num)
        self.num_per_cat = self.num_samples // cats_num if cats_num > 1 else self.num_samples
        mlog.info("Num per cat: %s", self.num_per_cat)
        self.rel_counter = {}
        self.rel_filter = rel_filter
        self.lang_counter = {}
        self.sel_rels = []
        if "other_rel" in ex_type:
            self.samples_per_head = 0
            self.num_per_cat = 0
            self.sel_rels = all_rels
            if "@" in ex_type:
                _rels = ex_type.split("@")[1]
                self.sel_rels = _rels.split("-")
        if rel_filter and not rel_filter in sel_rels:
            self.sel_rels.append(rel_filter)
        self.methods = method.split("+")
        self.sampling = sampling
        if len(self.methods) > 1 and split_name == "validation":
            self.methods = self.methods[0]

        self.old_input = ""
        self.si = 0
        self.example_counter = 0
        self.ex_df = pd.DataFrame()
        self._sels = self.sel_rels.copy()
        dlog.info("sels: %s", self._sels)
        self.save_path = save_ds_path  + "-".join(self.methods) + \
                "_" + str(len(split_df)) + "_" + str(self.num_samples) + ".pickle"
        if Path(self.save_path).is_file() and self.num_samples > 100_000 and not self.split_name == "sample":
            mlog.info("Loading from saved data %s ", self.save_path)
            self.load()

    def save(self):
        data = (self.flat_data, self.data_split)
        if not Path(self.save_path).exists():
            with open(self.save_path, "wb") as f:
                pickle.dump(data,f)
        else:
            mlog.info("The file already exists, skipping save ...")

    def load(self):
        with open(self.save_path, "rb") as f:
           data = pickle.load(f)
        self.flat_data, self.data_split = data
         
    def __iter__(self):
        iter_start = self.start
        iter_end = self.num_samples
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
             iter_start = self.start
             iter_end = self.num_samples
        else:  # in a worker process
             # split workload
             per_worker = int(math.ceil((self.num_samples - self.start) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = self.start + worker_id * per_worker
             iter_end = min(iter_start + per_worker, self.num_samples)
        if self.flat_data:
            _iter = iter(self.flat_data)
        else:
            _iter = iter(self.fill_data(iter_start, iter_end))
        self.example_counter = 0
        return map(self.preproc, _iter)

    def preproc(self, data):
        event = data[0]
        resp = data[1]
        extra = data[2]
        rel = extra["rel"]
        targ_col = extra["targ_col"]
        inp = extra["inp"]
        d = data[3]
        context_df = data[4]
        index = data[5]
        rel_token = atomic_relation_mappings[rel]
        if self.natural:
            resp = resp.replace("PersonX intends", "")
            resp = resp.replace("PersonX قصد دارد", "")
        resp = resp.strip()
        gen_token = gen_tokens[targ_col]
        input_lang = langs[inp]
        target_lang = langs[targ_col]
        mt = self.methods[0]
        if "-fa" in mt and input_lang == "fa":
            event = toPers(event)
        if "-fa" in mt and target_lang == "fa":
            resp = toPers(resp)

        qtemp, anstemp, ex_qtemp, ex_anstemp, context = create_templates(mt, 
                gen_pos="end", prompt_pos=self.prompt_pos)
        plen = relation_prompt_lengths[rel][0]
        if self.only_blanks and "___" in event:
            event = event.replace("___", "{ph}")
        mask = random.randint(0, plen-1)
        _qtemp = fill_consts(qtemp, ex_qtemp, context,d, context_df, mask=mask,method = mt)
        _anstemp = fill_consts(anstemp, ex_anstemp, context,d, context_df, mask=mask,method = mt)
        _query = fill_vars(_qtemp, rel, event, gen_token, resp, 
                input_lang, target_lang) 
        query = (index, _query)
        response = fill_vars(_anstemp, rel, event, gen_token, resp, 
                input_lang, target_lang)
        lang = input_lang + "2" + target_lang
        if self.example_counter < 10:
            clog.info(f"%%%%%%%%%%%%%%%%%% {lang} {mt} %%%%%%%%%%%%%%%%%%%")
            clog.info(inp + "====>" + targ_col)
            _q = _query.replace(">",">\n") 
            clog.info(input_lang + ":"+ _q)
            clog.info(target_lang + ":" + response)
            self.example_counter += 1
        #if not rel in self.data_split:
        #    self.data_split[rel] = {}
        #if not lang in self.data_split[rel]:
        #    self.data_split[rel][lang] = []
        #if query not in self.data_split[rel][lang]:
        #    self.data_split[rel][lang].append({query:[response]})
        #else:
        #    self.data_split[rel][lang][query].append(response)
        return (_query, response, rel, lang, index)

    def fill_all_data(self, iter_start, iter_end, show_progress=True):
        flat_data = []
        if iter_end < 0:
            iter_end = self.num_samples
        kk = 0 
        dlog.info("========================== SPLIT: %s", self.split_name)
        dlog.info("get data from %s to %s", iter_start, iter_end)
        dlog.info("total rows: %s", len(self.split_df))
        context_df = None
        if show_progress:
            pbar = tqdm(total = self.num_samples, position=0, leave=True) #,dynamic_ncols=True)
            pbar.set_description("Preparing iterator "+ self.split_name)

        for index, d in self.split_df.iterrows():
            if kk < iter_start:
                dlog.info("!!!!!!!!! before start %s", iter_start)
                kk += 1
                continue
            rel = d["prefix"]
            inp = "input_text"
            targ_col = "target_text"
            event = d[inp]
            resp = d[targ_col]
            lang = "en2en"
            if kk > iter_end:
                self.flat_data.extend(flat_data)
                return flat_data
            extra = {"inp":inp, "targ_col":targ_col, "rel":rel}
            flat_data.append((event, resp, extra, d, context_df, index))
            kk += 1
            if show_progress:
                pbar.update()
        self.flat_data.extend(flat_data)
        return flat_data

    def fill_data(self, iter_start, iter_end, show_progress=True):
        flat_data = []
        if iter_end < 0:
            iter_end = self.num_samples
        kk = 0 
        dlog.info("==========NNNNN========= SPLIT: %s", self.split_name)
        dlog.info("get data from %s to %s", iter_start, iter_end)
        dlog.info("total rows: %s", len(self.split_df))
        context_rows=[]
        context_df = None
        if show_progress:
            pbar = tqdm(total = self.num_samples, position=0, leave=True) #,dynamic_ncols=True)
            pbar.set_description("Preparing iterator "+ self.split_name)

        for index, d in self.split_df.iterrows():
            if kk < iter_start:
                dlog.info("!!!!!!!!! before start %s", iter_start)
                kk += 1
                continue
            rel = d["prefix"]
            if not rel in self.rel_counter:
                self.rel_counter[rel] = 0
            dlog.info("rel counter %s", self.rel_counter)
            if self.num_per_cat > 0 and self.rel_counter[rel] > self.num_per_cat:
                dlog.info("!!!!!!!!! number per cat limit reached %s for %s", rel, self.num_per_cat)
                continue 
            if "other_rel" in self.ex_type:
                if len(context_rows) >= len(self.sel_rels):
                    context_df = pd.DataFrame(data=context_rows)
                    self.ex_df = self.ex_df.append(context_df)
                    if self.rel_filter:
                        for item in context_rows:
                            if item["prefix"] == self.rel_filter:
                                d = item
                                rel = d["prefix"]
                    context_rows = []
                    self._sels = self.sel_rels.copy()
                else:
                    if (rel in self._sels and d["target_text"] != "none"): 
                        context_rows.append(d)
                        self._sels.remove(rel)
                    dlog.info("!!!!!!!!! just for including in conext rows %s", len(context_rows))
                    continue
            elif self.ex_type == "same_rel":
                context_df = self.split_df[self.split_df["prefix"] == rel].sample(n=self.sampling)
            elif self.ex_type:
                raise ValueError("Ex_type is invalid:" + self.ex_type)
            eng_inp = d["input_text"]
            self.si += 1
            if eng_inp != self.old_input:
                context_rows = []
                self._sels = self.sel_rels.copy()
                self.old_input = eng_inp
                self.si = 0
            elif self.samples_per_head > 0 and self.si > self.samples_per_head:
                dlog.info("!!!!!!!!! samples per head limit %s", self.samples_per_head)
                continue
            for inp in inputs:
                if not inp in d or len(d[inp]) <= 1:
                    dlog.info("!!!!!!!!! not in dataset %s", inp)
                    continue
                if self.inp_include and not any(x in inp for x in self.inp_include.split("|")):
                    dlog.info("!!!!!!!!! not included input col %s", self.inp_include)
                    continue
                if self.inp_exclude and any(x in inp for x in self.inp_exclude.split("|")):
                    dlog.info("!!!!!!!!! excluded input col %s", self.inp_exclude)
                    continue
                input_lang = langs[inp]
                self.rel_counter[rel] += 1
                for targ_col in targets:
                    if not targ_col in d or len(d[targ_col]) <= 1:
                        dlog.info("!!!!!!!!! not target lang %s", targ_col)
                        continue
                    if self.targ_include and not any(x in targ_col for x in self.targ_include.split("|")):
                        dlog.info("!!!!!!!!! not included target col %s", self.targ_include)
                        continue
                    if self.targ_exclude and any(x in targ_col for x in self.targ_exclude.split("|")):
                        dlog.info("!!!!!!!!!  target exclude %s", self.targ_exclude)
                        continue
                    event = d[inp]
                    resp = d[targ_col]
                    target_lang = langs[targ_col]
                    lang = input_lang + "2" + target_lang
                    if not lang in self.lang_counter:
                        self.lang_counter[lang] = 1
                    else:
                        self.lang_counter[lang] += 1
                    if (self.lang_counter[lang] > self.num_records 
                        or self.lang_counter[lang] > iter_end):
                        dlog.info("Lang limit reached! %s %s", lang, self.lang_counter[lang])
                        self.flat_data.extend(flat_data)
                        return flat_data
                    extra = {"inp":inp, "targ_col":targ_col, "rel":rel}
                    flat_data.append((event, resp, extra, d, context_df, index))
                    if show_progress:
                        pbar.update()
                    kk += 1
                    if (kk > iter_end or kk > self.num_records):
                        dlog.info("record limit reached!")
                        self.flat_data.extend(flat_data)
                        return flat_data
            
        self.flat_data.extend(flat_data)
        return flat_data

def save_data(ex_df, save_ds_path):
    if save_ds_path and len(ex_df) > 0:
        ex_df = ex_df.drop_duplicates(["input_text","prefix"])
        ex_df = ex_df.sort_values(by=["input_text","prefix"])
        mlog.info("DF saved as %s", save_ds_path)
        ex_df.to_csv(save_ds_path, index=False, sep="\t")

def save_checkpoint(model, tokenizer, optimizer, scheduler, step, 
                   best_eval_step, best_dev_loss, save_path):
    if "temp" in save_path:
        mlog.info("Saves in temp are skipped ")
        return

    mlog.info("Saving model ... %s", save_path)
    with open(save_path + "/best_model.txt", "a") as f:
        print("best_step:", best_eval_step, file=f)
        print("best dev loss:", best_dev_loss, file=f)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    #torch.save(model.state_dict(), os.path.join(save_path,"state_dict"))
    #torch.save(model, os.path.join(save_path,"mymodel"))
#    torch.save({
#            'step': step,
#            'eval_step': best_eval_step,
#            'model_state_dict': model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'scheduler_state_dict': scheduler.state_dict(),
#            }, os.path.join(save_path, "saved_states"))
#

