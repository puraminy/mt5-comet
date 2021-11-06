from pathlib import Path
from comet.utils.myutils import *
from sentence_transformers import SentenceTransformer, util
from transformers import AddedToken 
from sentence_transformers import CrossEncoder
import pandas as pd
from comet.transformers_ptuning import PTuningWrapper
from comet.transformers_ptuning.ptuning_wrapper import LSTMEmbeddingPromptEncoder, EmbeddingPromptEncoder
from rouge import Rouge
from tqdm import tqdm
import logging, sys
import re
import os
import torch
import json
from os.path import expanduser
home = expanduser("~")
if "ahmad" in home:
    logPath = "/home/ahmad/logs/"
    resPath = "/home/ahmad/logs/"
else:
    logPath = "/content/"
    resPath = "/content/drive/MyDrive/backup/results"

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


for logger, fname in zip([mlog,dlog,clog,vlog], ["main","data","cfg","eval"]):
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
end_token = "<extar_id_1>"
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

    added_tokens = [ 
            AddedToken(f"<{rel}_{i}>",lstrip=True,
                rstrip=False)
            for i in 
                range(enc_plen + dec_plen + 1)
    ]
    tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
    model.resize_token_embeddings(len(tokenizer))
    wrapped_model = PTuningWrapper(model,prompt_encoder,decoder_prompt_encoder,prompt_token_fn=get_prompt_token_fn(id_offset,enc_plen + dec_plen))
    if prompt_path:
        wrapped_model.prompt_encoder.load(prompt_path)

    return wrapped_model

def format_temp(template, rel, event, gen_token, resp, lang):
    rel_token = atomic_relation_mappings[rel]        
    enc_token = encoder_relation_mappings[rel] if rel in encoder_relation_mappings else ""
    dec_token = decoder_relation_mappings[rel] if rel in encoder_relation_mappings else ""
    rel_natural = relation_natural_mappings[rel][lang]        
    return template.format(event=event, 
                         response=resp,
                         rel=rel, 
                         enc_token=enc_token, 
                         dec_token=dec_token, 
                         rel_token=rel_token,
                         rel_natural=rel_natural,
                         gen=gen_token,
                         ph=placeholder_token,                                                                       end=end_token)

#%% Aggregate instances of queries and corresponding responses
# (str)split_name -> (dict) query -> (list) response 
def gen_resp(model, tokenizer, query, gen_token = "", gen_param = "greedy"):
    if gen_param == "greedy":
        generation_params = {
            "max_length":80,
            "early_stopping":True,
            "num_beams":5,
            "num_return_sequences":3,
        }
    elif gen_param == "top_p":
        generation_params = {
            "do_sample":True, 
            "top_p":0.95, 
            "num_return_sequences":3, 
            "repetition_penalty":2.5,
            "max_length":80,
        }
    inputs = tokenizer(query,return_tensors='pt').to(device=device)
    if False: #gen_token != "":
        gen_token_id = tokenizer.convert_tokens_to_ids(gen_token)
        hyps = model.generate(**inputs,**generation_params,
                decoder_start_token_id=gen_token_id)
        hyps = tokenizer.batch_decode(hyps,skip_special_tokens=True)
    else:
        hyps = model.generate(**inputs,**generation_params)
        hyps = tokenizer.batch_decode(hyps,skip_special_tokens=True)
    return hyps

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
def fill_data(split_df, split_name, inputs, targets, qtemp, anstemp, 
            num_samples=0, 
            ignore_blanks=False,
            include="",
            exclude="",
            pred_tresh=0,
            nli_group="all"): 
    dlog.info("building query responses for {}".format(split_name))
    dlog.info(f"len:{len(split_df)}")
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
            if "{event_en}" in qtemp and input_lang != "en":
                continue
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
                if (("{response_en}" in anstemp or "{response_en}" in qtemp) 
                    and target_lang != "en"):
                    continue
                query = format_temp(qtemp, rel, event, gen_token, resp, input_lang) 
                resp = format_temp(anstemp, rel, event, gen_token, resp, target_lang)
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
                    data_split[rel][lang].append({query:[resp]})
                    if jj < 3:
                        dlog.info("Q:"+ query)
                        dlog.info("R:"+ resp)
                else:
                    data_split[rel][lang][query].append(resp)
                flat_data.append((query, resp))
                kk += 1
            #didn't convert ___ to <blank>
            #didn't normalize to lowercase
    return data_split, flat_data, kk

def save_checkpoint(model, optimizer, scheduler, step, 
                   best_eval_step, best_dev_loss, save_path):
    model.save_pretrained(save_path)
    torch.save({
            'step': step,
            'eval_step': best_eval_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(save_path, "saved_states"))

    with open(save_path + "/best_model.txt", "a") as f:
        print("best_step:", best_eval_step, file=f)
        print("best dev loss:", best_dev_loss, file=f)

def bert_score(bert_scorer, hyps, refs):
        if bert_scorer == None:
            return 0, 0, 0.0

        embeddings1 = bert_scorer.encode(hyps, device=device, convert_to_tensor=True)
        embeddings2 = bert_scorer.encode(refs, device=device, convert_to_tensor=True)

        #Compute cosine-similarities for each sentence with each other sentence
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

        #Find the pairs with the highest cosine similarity scores
        pairs = []
        rows = cosine_scores.shape[0]
        cols = cosine_scores.shape[1]
        for i in range(rows):
            for j in range(cols):
                pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
            #logging.info({'index': [i, j], 'score': cosine_scores[i][j]})

        #Sort scores in decreasing order
        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

        top = pairs[0]
        best_hyp_index = top["index"][0]
        best_ref_index = top["index"][1]

        return best_hyp_index, best_ref_index, top["score"] 
# vvvvvvvvvvvvvvv
# ################################### Evaluation #########################
def eval(model, tokenizer, val_data, interactive, save_path, output_name, val_records, gen_param="greedy"):  
    base_path = "/content/drive/MyDrive/pret"
    if "ahmad" in home:
        base_path = "/home/ahmad/pret"

    local_path = f"{base_path}/paraphrase-multilingual-MiniLM-L12-v2"        
    if not Path(local_path).exists():
        local_path = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    bert_scorer = None #SentenceTransformer(local_path)
    rouge_scorer = Rouge()
    local_path = f"{base_path}/nli-roberta-base"
    if not Path(local_path).exists():
        local_path = 'sentence-transformers/nli-roberta-base'
    nli_model = None #CrossEncoder(local_path)
    nli_counter = {}
    for l in nli_map:
        nli_counter[l] = 0
    #df = df.groupby(['prefix','input_text'],as_index=False)[target].agg({"target_text":'<br />'.join})
    #resp_const_parts = re.split("{.*}", anstemp)
    resp_const_parts = ["<extra_id_0>", "<extra_id_1>", "."]
    mlog.info("Scoring...")
    model.eval()
    pbar = tqdm(total = val_records)
    rows = []
    counter = {"all":0}
    sum_bert = {} 
    sum_rouge = {}
    mean_bert = {}
    mean_rouge = {}
    hyp_counter = [0]*5
    for rel in val_data.keys():
        vlog.info(f"%%%%%%%%%%%%%%%%%%%%%%%%% {rel} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        for lang in val_data[rel].keys():
            vlog.info(f"%%%%%%%%%%%%%%%%%%%%%%%%%% { lang } %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            for queries in val_data[rel][lang]:
                for query, tails in queries.items():
                    data = {}
                    scope = rel + "_" + lang
                    if not scope in sum_bert: 
                        sum_bert[scope] = 0
                        sum_rouge[scope] = 0
                        counter[scope] = 0
                    vlog.debug("&&&&&&&&&&&&&&&&& All Targets &&&&&&&&&&&&&&")
                    for _tail in tails:
                        vlog.debug(_tail)
                    vlog.debug("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                    if interactive: #interactive mode
                        query = get_input("Enter an even or Enter) skip, c) continue, e) exit.")
                        resp = "NA"
                        if query == "e":
                            return data_split, flat_data, kk
                        if query == "c":
                            interactive = False
                    gen_token = gen_tokens[lang]
                    hyps = gen_resp(model, tokenizer, query, gen_token, gen_param)
                    input_text = re.sub(r'<.*?>','',query)
                    top_hyp = hyps[0]
                    for const in resp_const_parts:
                        top_hyp = top_hyp.replace(const, "")
                    if not top_hyp.strip():
                        top_hyp = "EMPT"
                    new_tails = []
                    for tail in tails:
                        if not tail.strip():
                            continue
                        nt = tail
                        for const in resp_const_parts:
                            nt = nt.replace(const,"")
                        new_tails.append(nt)
                    tails = new_tails
                    data["input_text"] = input_text 
                    data["target_text"] = "<br />".join(tails)
                    data["pred_text1"] = top_hyp
                    data["prefix"] = rel
                    data["langs"] = lang
                    #Compute embeddings
                    hi, ri, cur_score = bert_score(bert_scorer, hyps, tails)
                    best_hyp = hyps[hi]
                    best_ref = tails[ri]
                    hyp_counter[hi] += 1
                    if nli_model:
                        pair = (best_hyp, best_ref)
                        nli_scores = nli_model.predict(pair)  
                        _max  = nli_scores.argmax()
                        label = nli_map[_max]
                        nli_counter[label] += 1
                        data["nli_group"] = label
                        vlog.info("Label:"+ label)
                    data["top"] = best_ref
                    data["all_preds"] = "<br />".join(hyps) 
                    data["top_pred"] = best_hyp
                    data["bert_score"] = float("{:.2f}".format(cur_score))
                    rows.append(data)
                    sum_bert[scope] += cur_score
                    counter[scope] += 1
                    counter["all"] += 1
                    mean_bert[scope] = "{:.4f}".format(sum_bert[scope] / counter[scope])
                    #tqdm.write(f"Mean score:{mean_bert}")
                    vlog.info("")
                    vlog.info(str(counter["all"])+ ":"+query)
                    vlog.info("Prediction:"+ top_hyp)
                    vlog.info("Closest tail:"+ best_ref)
                    

                    mlog.debug(f"TOP hyp:{top_hyp}")
                    mlog.debug(f"Tails: {tails}")
                    rouge_score = rouge_scorer.get_scores(top_hyp, ".".join(tails), avg=True, ignore_empty=True)
                    rouge_score = rouge_score["rouge-l"]["f"]
                    data["rouge_score"] = rouge_score
                    sum_rouge[scope] += rouge_score
                    mean_rouge[scope] = "{:.4f}".format(sum_rouge[scope] / counter[scope])
                    vlog.info("Bert Score:{:.4f}--{}".format(cur_score, mean_bert[scope]))
                    vlog.info("Rouge Score:{:.4f}--{}".format(rouge_score, mean_rouge[scope]))
                    vlog.info("------------------------------------------------------")
                    pbar.set_description(f"{scope} :Bert:{mean_bert[scope]} Rouge {mean_rouge[scope]} ")
                    pbar.update(1)

    # %%%%%%%%%%%%%%%%%%
    new_df = pd.DataFrame(rows)
    new_df = new_df[new_df["bert_score"] > 0]
    pbar.close()
    out1 = os.path.join(save_path,f"scored_{output_name}.tsv")
    out2 = os.path.join(resPath,f"scored_{output_name}.tsv")
    out3 = os.path.join(logPath,f"scored_{output_name}.tsv")

    new_df.to_csv(out1, sep="\t", index=False)
    new_df.to_csv(out2, sep="\t", index=False)
    new_df.to_csv(out3, sep="\t", index=False)
    pred_counts = new_df['pred_text1'].unique()

    mean_bert_str = json.dumps(mean_bert, indent=2)
    mean_rouge_str = json.dumps(mean_rouge, indent=2)
    res = {}
    res["rouge"] = mean_rouge
    res["bert"] = mean_bert
    res["distinct"] = len(pred_counts)
    res["hyps"] = hyp_counter

    dictPath(output_name, results, res, sep="_")
    with open(os.path.join(resPath, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(logPath, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    for logger in [mlog, vlog, clog]:
        logger.info("Len data frame: {}".format(len(new_df)))
        logger.info("Rouge:{} BERT {} ".format(mean_rouge_str, mean_bert_str))
        logger.info("DF mean Bert Score: {}".format(new_df["bert_score"].mean()))
        logger.info("DF mean Rouge Score: {}".format(new_df["rouge_score"].mean()))
        logger.info("nli_counter: {}".format(nli_counter))
        logger.info("hyp_counter: {}".format(hyp_counter))
        logger.info("Distinct preds:{}".format(len(pred_counts)))

