from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers import AddedToken 
from sentence_transformers import CrossEncoder
import pandas as pd
from comet.transformers_ptuning import PTuningWrapper
from comet.transformers_ptuning.ptuning_wrapper import LSTMEmbeddingPromptEncoder, EmbeddingPromptEncoder
from comet.evaluation.rouge.rouge import Rouge
from tqdm import tqdm
import logging, sys
import re
import os
import torch
from os.path import expanduser
home = expanduser("~")
if "ahmad" in home:
    logPath = "/home/ahmad/logs/"
else:
    logPath = "/content/"
logFilename = os.path.join(logPath, "all.log") #app_path + '/log_file.log'
# log only important messages
logging.basicConfig(filename=logFilename)
consoleHandler = logging.StreamHandler()
mlog = logging.getLogger("comet.main")
mlog.setLevel(logging.INFO)
logFilename = os.path.join(logPath, "main.log") #app_path + '/log_file.log'
mHandler = logging.FileHandler(logFilename)
mlog.addHandler(mHandler)
mlog.addHandler(consoleHandler)

clog = logging.getLogger("comet.cfg")
logFilename = os.path.join(logPath, "cfg.log") #app_path + '/log_file.log'
cHandler = logging.FileHandler(logFilename)
clog.addHandler(cHandler)
clog.setLevel(logging.INFO)

dlog = logging.getLogger("comet.data")
logFilename = os.path.join(logPath, "data.log") #app_path + '/log_file.log'
dHandler = logging.FileHandler(logFilename)
dlog.addHandler(dHandler)
dlog.setLevel(logging.INFO)

vlog = logging.getLogger("comet.eval")

logFilename = os.path.join(logPath, "eval.log") #app_path + '/log_file.log'
vHandler = logging.FileHandler(logFilename)
vlog.addHandler(vHandler)
vlog.setLevel(logging.INFO)


device = 'cuda'
def set_device(dev):
    global device
    device = dev

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
generation_params = {
    "max_length":80,
    "early_stopping":True,
    "num_beams":5,
    "num_return_sequences":3,
}
def gen_resp(model, tokenizer, query, gen_token = ""):
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
            natural=False,
            pred_tresh=0,
            nli_group="all"): 
    dlog.info("building query responses for {}".format(split_name))
    dlog.info(f"len:{len(split_df)}")
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
    if pred_tresh > 0 and "pred1_score" in split_df:
        split_df = split_df[split_df["pred1_score"] > pred_tresh]
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
            if ((natural and not "natural" in inp) or 
                (not natural and "natural" in inp)):
                continue
            input_lang = langs[inp]
            for targ_col in targets:
                if not targ_col in d or len(d[targ_col]) <= 1:
                    continue
                if ((natural and not "natural" in targ_col) or 
                    (not natural and "natural" in targ_col)):
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
def eval(model, tokenizer, val_data, num_generations, 
        interactive, save_path, verbose):  

    base_path = "/content/drive/MyDrive/pret"
    if "ahmad" in home:
        base_path = "/home/ahmad/pret"
    if verbose:
        vlog.addHandler(StreamHandler)

    local_path = f"{base_path}/paraphrase-multilingual-MiniLM-L12-v2"        
    if not Path(local_path).exists():
        local_path = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    bert_scorer = SentenceTransformer(local_path)
    rouge_scorer = Rouge()
    local_path = f"{base_path}/nli-roberta-base"
    if not Path(local_path).exists():
        local_path = 'sentence-transformers/nli-roberta-base'
    nli_model = CrossEncoder(local_path)
    nli_counter = {}
    for l in nli_map:
        nli_counter[l] = 0
    #df = df.groupby(['prefix','input_text'],as_index=False)[target].agg({"target_text":'<br />'.join})
    #resp_const_parts = re.split("{.*}", anstemp)
    resp_const_parts = ["<extra_id_0>", "<extra_id_1>"]
    mlog.info("Scoring...")
    model.eval()
    sum_bert = 0 
    sum_rouge = 0
    total = num_generations
    #if num_generations == 0:
    total = num_generations
    max_score = 0
    pbar = tqdm(total = total)
    rows = []
    old_input = ""
    ii = 0
    hyp_counter = [0]*5
    for rel in val_data.keys():
        vlog.info(f"%%%%%%%%%%%%%%%%%%%%%%%%% {rel} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        for lang in val_data[rel].keys():
            vlog.info(f"%%%%%%%%%%%%%%%%%%%%%%%%%% { lang } %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            if ii > num_generations:
                break
            for queries in val_data[rel][lang]:
                for query, tails in queries.items():
                    data = {}
                    if verbose:
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
                    hyps = gen_resp(model, tokenizer, query, gen_token)
                    input_text = re.sub(r'<.*?>','',query)
                    top_hyp = hyps[0]
                    if top_hyp == "":
                        top_hyp = "."
                    for const in resp_const_parts:
                        top_hyp = top_hyp.replace(const, "")
                    new_tails = []
                    for tail in tails:
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
                    data["pred1_score"] = float("{:.2f}".format(cur_score))
                    rows.append(data)
                    sum_bert += cur_score
                    ii += 1
                    mean_bert = "{:.4f}".format(sum_bert / ii)
                    #tqdm.write(f"Mean score:{mean_bert}")
                    vlog.info("")
                    vlog.info(str(ii)+ ":"+query)
                    vlog.info("Prediction:"+ top_hyp)
                    vlog.info("Closest tail:"+ best_ref)

                    rouge_score = rouge_scorer.calc_score(candidate=[top_hyp], refs=tails)
                    data["rouge_score"] = rouge_score
                    sum_rouge += rouge_score
                    mean_rouge = "{:.4f}".format(sum_rouge / ii)
                    vlog.info("------------------------------------------------------")
                    pbar.set_description(f"Bert:{mean_bert} Rouge {mean_rouge} ")
                    pbar.update(1)

    # %%%%%%%%%%%%%%%%%%
    new_df = pd.DataFrame(rows)
    new_df = new_df[new_df["pred1_score"] > 0]
    pbar.close()
    out = os.path.join(save_path,"scored_results.tsv")
    out2 = os.path.join(logPath,"scored_results.tsv")
    new_df.to_csv(out, sep="\t", index=False)
    new_df.to_csv(out2, sep="\t", index=False)
    for logger in [mlog, vlog, clog]:
        logger.info("Len data frame: {}".format(len(new_df)))
        logger.info(f"Bert:{mean_bert} Rouge {mean_rouge} ")
        logger.info("DF mean Bert Score: {}".format(new_df["pred1_score"].mean()))
        logger.info("DF mean Rouge Score: {}".format(new_df["rouge_score"].mean()))
        logger.info("nli_counter: {}".format(nli_counter))
        logger.info("hyp_counter: {}".format(hyp_counter))
        pred_counts = new_df['pred_text1'].unique()
        logger.info("Distinct preds:{}".format(len(pred_counts)))

