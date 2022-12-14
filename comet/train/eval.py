
from itertools import islice
from comet.train.common import *
import nltk
import math
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
from sklearn.metrics import f1_score
from rouge import Rouge
#%% Aggregate instances of queries and corresponding responses
# (str)split_name -> (dict) query -> (list) response 

from datasets import load_metric
import nltk


def forward_step(model, batch, no_model_batch, accumulation_tiny_steps=1, mode="train", task_ids=None):
    for k in no_model_batch:
        if k not in  ["resp", "query", "target", "wrap", "freeze", "unfreeze", "method", "task"]:
            no_model_batch[k] = no_model_batch[k].to(device)
    #if task_ids is not None:
    #    result =  model.forward(task_ids, add_prior=True, **batch)
    #else:
    result = model(**batch)
    logits = result["logits"]
    forw_out = {
        "logits": logits
    }
    if "loss" in result: # and not "loss_mask" in no_model_batch:
        loss = result['loss']/accumulation_tiny_steps
    else:
        losses = torch.nn.functional.cross_entropy(
            result['logits'].reshape(-1,result['logits'].size(2)),
            no_model_batch['labels'].reshape(-1,),
            reduction='none'
        ).reshape(result['logits'].size(0),-1)
        if "loss_mask" in no_model_batch:
            loss_mask = no_model_batch["loss_mask"]
            #loss_mask = loss_mask.to(device)
            losses = (losses * loss_mask).sum(-1) / loss_mask.sum(-1)
            loss = losses.mean()
        else:
            loss = losses.mean()
        forw_out["loss_batch"] = losses

    forw_out["loss"] = loss

    return forw_out


def evaluate1(tokenizer, eval_data_loader, model, device, seed =0, mode="dev", save_path="", wrap=True, task_ids=None):
    """Evaluation."""
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    step = 0
    #set_random_seed(seed)

    all_idx = []
    all_preds = []
    all_labels = []
    all_gens = []
    all_resps = []
    all_queries = []
    gen_model = model.underlying_model if wrap else model
    with torch.no_grad():
        for model_batch, no_model_batch in eval_data_loader:
            for k in model_batch:
                model_batch[k] = model_batch[k].to(device)
            for k in no_model_batch:
                if k not in  ["resp", "query", "target", "wrap", "freeze", "unfreeze", "method"]:
                    no_model_batch[k] = no_model_batch[k].to(device)

            decs = generate(gen_model, tokenizer, model_batch, task_ids)
            all_gens.extend(decs)

            forw_out = forward_step(model, model_batch, no_model_batch, mode="test")
            loss = forw_out["loss"].item() if "loss" in forw_out else 0
            total_loss += loss

            logits_list = forw_out["logits"]
            seq_len = logits_list.size()[1]
            seq_preds = []
            for i in range(seq_len):
                pred_token_logits = logits_list[:, i, :]
                preds = torch.argmax(pred_token_logits, dim=-1)
                seq_preds.append(preds.tolist())
            _seq_preds = list(zip(*seq_preds))
            all_preds.extend(_seq_preds)

            if "idx" in no_model_batch: 
                gathered_idx = no_model_batch["idx"]
                all_idx.extend(gathered_idx)

            #labels = no_model_batch["labels"][:, 1]
            # my code
            labels = model_batch["labels"]#[:, 1]
            gathered_labels = labels.tolist() 
            all_labels.extend(gathered_labels)

            all_queries.extend(no_model_batch["query"])
            all_resps.extend(no_model_batch["resp"])

            step += 1

    total_loss /= step

    #all_idx = torch.cat(all_idx, dim=0).cpu().tolist()
    #all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
    #all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
    preds_decs = []
    for p in all_preds:
        dec = tokenizer.convert_ids_to_tokens(p)
        preds_decs.append(dec)
    labels_decs = []
    for l in all_labels:
        l = [0 if x == -100 else x for x in l] 
        dec = tokenizer.convert_ids_to_tokens(l)
        labels_decs.append(dec)
    _preds = []
    _labels = []
    _gens = []
    c = 0
    i = 0
    inps = 0
    inp = ""
    rows = []
    for p,l,r, g, q in zip(preds_decs, labels_decs, all_resps, all_gens, all_queries):
        _gens.append(g)
        _preds.append(p[1].lower())
        _labels.append(l[1].lower())
        resp = re.sub(r'<.*?>','', r)
        resp = resp.strip()
        if q != inp:
            inps +=1
            inp = q
        dd = {"top": resp, "top_pred": g}
        rows.append(dd)
        print("-"*80)
        print("{}) {}".format(i, q))
        print("")
        if any(x in resp.split() for x in g.split()):
            c +=1
        print(" "*10,"True:",r, " | ", resp)
        print(" "*10,"Pred:",p)
        print(" "*10,"Gen:",g)
        i += 1
    acc1 = c/i
    acc2 = c/inps
    print("{:.2f} = {}/{} | {:.2f} = {}/{}".format(acc1,c,i, acc2, c, inps))
    batch = pd.DataFrame(data=rows)
    batch.to_csv(os.path.join(save_path, "{:.3f}".format(acc1)+".tsv"), sep="\t", index=False)
    #st_score = run_sts_benchmark(batch, st_embed)
    metric_list = ["rouge", "meteor", "bertscore"]
    metric_list = ["bertscore"]
    #summary = calc_metrics(batch["top_pred"].tolist(), batch["top"].tolist(), metric_list)
    bscore = 0.0 #summary["bertscore_f1"]
    #eval_metric = acc_f1_metric
    eval_metric = acc_metric
    res = eval_metric(tokenizer, _preds, _labels, save_path=save_path)
    print(res)

    return acc1, acc2, bscore, total_loss 




#  the code below refers to the https://github.com/Yale-LILY/FeTaQA/blob/main/end2end/train.py
def postprocess_text(preds, labels, metric_name):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    if metric_name == "rouge":
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    elif metric_name == "sacrebleu":  # sacrebleu
        labels = [[label] for label in labels]
    elif metric_name == "bleu":
        preds = [pred.split(' ') for pred in preds]
        labels = [[label.split(' ')] for label in labels]
    else:
        pass

    return preds, labels

device = "cpu"
def set_device(dev):
    global device
    device = dev


from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet

lemmatizer = nltk.WordNetLemmatizer()
#nltk.download('averaged_perceptron_tagger')
#word tokenizeing and part-of-speech tagger
def get_verb(document):
    tokens = [nltk.word_tokenize(sent) for sent in [document]]
    postag = [nltk.pos_tag(sent) for sent in tokens][0]
    for item in postag:
        w,t = item
        if t in ["V","VB", "VBD"]:
            return w
    return ""

def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

def generate(model, tokenizer, batch, gen_token = "", gen_param = "greedy", at_mask=None, task_ids=None):
    skip_special = "True"
    #verb = get_verb(query)
    #vlog.info("Ignoring verb %s", verb)
    bad_words_ids = None
    #extra_id_0 = tokenizer.convert_tokens_to_ids(["<extra_id_0>"])[0]
    #if verb:
    #    bad_words_ids = tokenizer(verb).input_ids
    if "@" in gen_param:
        gen_param, skip_special = gen_param.split("@")
    if gen_param == "greedy":
        gen_kwargs = {
            "max_length":40,
            "num_beams":5,
            "repetition_penalty":5.5,
            "num_return_sequences":1,
            "bad_words_ids": bad_words_ids
        }
    elif gen_param == "top_p" or gen_param == "top_k":
        gen_kwargs = {
            "max_length":40,
            "do_sample":True, 
            "top_p":0.9, 
            "top_k":10,
            "num_beams":5,
            "temperature": 1.0,
            "num_return_sequences":1, 
            "repetition_penalty":5.5,
            "bad_words_ids": bad_words_ids
        }
    #batch.to(device)
    #if "labels" in batch:
    #    gen_kwargs["labels"] = batch["labels"]
    if "description_input_ids" in batch:
        gen_kwargs["description_input_ids"] = batch["description_input_ids"]
    if "description_attention_mask" in batch:
        gen_kwargs["description_attention_mask"] = batch["description_attention_mask"]
    if "knowledge_input_ids" in batch:
        gen_kwargs["knowledge_input_ids"] = batch["knowledge_input_ids"]
    if "knowledge_attention_mask" in batch:
        gen_kwargs["knowledge_attention_mask"] = batch["knowledge_attention_mask"]
    if "task_ids" in batch:
        gen_kwargs["task_ids"] = batch["task_ids"]

    input_batch = {}
    input_batch["input_ids"] = batch["input_ids"]
    input_batch["attention_mask"] = batch["attention_mask"]
    input_ids, attention_mask = trim_batch(**input_batch, pad_token_id=tokenizer.pad_token_id)
    decs = []
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    if False: #gen_token != "":
        gen_token_id = tokenizer.convert_tokens_to_ids(gen_token)
        hyps = model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
                decoder_start_token_id=gen_token_id)
        hyps = tokenizer.batch_decode(hyps,skip_special_tokens=False)
    else:
        #breakpoint()
        if False: #task_ids is not None:
            hyps = model.generate(task_ids, 
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                    )
        else:
            hyps = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                    )
        hyps = tokenizer.batch_decode(hyps,skip_special_tokens=skip_special == "True")
        decs.extend(hyps)
    return decs
# ggggggggg
def batch_generate(model, tokenizer, queries, batch_size=5, gen_token = "", gen_param = "greedy", at_mask=None):
    skip_special = "True"
    #verb = get_verb(query)
    #vlog.info("Ignoring verb %s", verb)
    bad_words_ids = None
    #if verb:
    #    bad_words_ids = tokenizer(verb).input_ids
    if "@" in gen_param:
        gen_param, skip_special = gen_param.split("@")
    if gen_param == "greedy":
        gen_kwargs = {
            "max_length":160,
            "num_beams":5,
            "repetition_penalty":5.5,
            "num_return_sequences":1,
            "bad_words_ids": bad_words_ids
        }
    elif gen_param == "top_p":
        gen_kwargs = {
            "max_length":160,
            "do_sample":True, 
            "top_p":0.9, 
            "top_k":10,
            "num_beams":5,
            "temperature": 1.0,
            "num_return_sequences":1, 
            "repetition_penalty":5.5,
            "bad_words_ids": bad_words_ids
        }
    with torch.no_grad():
        examples = queries
        decs = []
        for batch in list(chunks(queries, batch_size)):
            batch = tokenizer(batch, return_tensors="pt", max_length=200, truncation=True, padding=True).to(device)
            input_ids, attention_mask = trim_batch(**batch, pad_token_id=tokenizer.pad_token_id)

            if False: #gen_token != "":
                gen_token_id = tokenizer.convert_tokens_to_ids(gen_token)
                hyps = model.generate(input_ids=input_ids,**gen_kwargs,
                        attention_mask=attention_mask,
                        decoder_start_token_id=gen_token_id)
                hyps = tokenizer.batch_decode(hyps,skip_special_tokens=False)
            else:
                hyps = model.generate(input_ids=input_ids,**gen_kwargs,
                        attention_mask=attention_mask)
                hyps = tokenizer.batch_decode(hyps,skip_special_tokens=skip_special == "True")
                decs.extend(hyps)
    return decs

def bert_score(bert_scorer, hyps, refs):
        if bert_scorer == None:
            return 0, 0, 0.0

        hyps = [p.strip() for p in hyps]
        refs = [g.strip() for g in refs]

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

# ################################### Evaluation #########################
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

import debugpy
# vvvvvvvvvvvvvvv

def evaluate(test_set, dataloader, save_path, exp_info, val_records, gen_param="greedy", scorers="rouge", batch_size="20@5", model = None, tokenizer = None, preds_file = "", set_name = "test", rewrite_info = False, stop_level=0, seed=0, task_ids=None):  
    if rewrite_info:
        save_path = os.path.join(save_path, "full_results.tsv")
        if Path(save_path).is_file() and rewrite:
            df = pd.read_table(save_path)

        for key, info in exp_info.items():
            df[key] = info

        mlog.info("Saving results %s", save_path)
        df.to_csv(save_path, index=False, sep="\t")
        return


    mlog.info("Loading models for evaluation ..")
    mlog.info("%s", save_path)
    #set_random_seed(seed)

    #local_path = f"{base_path}/paraphrase-multilingual-MiniLM-L12-v2"        
    #df = df.groupby(['prefix','input_text'],as_index=False)[target].agg({"target_text":'<br />'.join})
    #resp_cost_toks = re.split("{.*}", anstemp)
    resp_cost_toks = ["<pad>","<extra_id_0>", "<extra_id_1>", "<extra_id_2>", "</s>", "."]
    if model is not None: model.eval()
    rows = []
    sel_rows = []
    mlog.info("Scoring...")
    pbar = tqdm(total=len(test_set), position=0, leave=True) #,dynamic_ncols=True)
    step = 0
    if "@" in batch_size:
        bs, gen_bs = batch_size.split("@")
    else:
        bs = batch_size
        gen_bs = max(2, bs - 5)
    bs = int(bs)
    gen_bs = int(gen_bs)
    vlog.disabled = True
    exit_loop = False
    nones = 0
    lang = "en2en"
    sel_inps = {}
    if preds_file:
        mlog.info("extention %s, %s", preds_file, Path(preds_file).suffix) 
        if Path(preds_file).suffix == ".json":
            with open(preds_file) as json_file:
                records = json.load(json_file)
            lines = []
            test_set = []
            for item in records:
                d = (item["text_in"], item["question"], item["answer_text"], item["meta"],item["id"], 0)
                test_set.append(d)
                lines.append(item["prediction"])
        else:
            with open(preds_file, 'r') as infile:
                  lines = infile.readlines()
            lines = lines[1:]
    l_count = 0
    ignore_special_tokens = False
    if "@" in gen_param:
        _, ist = gen_param.split("@")
        ignore_special_tokens = ist == "True"
    test_iter = iter(test_set)
    batches = batched(list(test_iter), bs)
    if model is not None:
        dl_iter = iter(dataloader)
    iid  = 0
    old_query = ""
    for batch_list in batches: 
        if exit_loop:
            break
        if model is not None:
            if False:
                queries = [x[0] for x in batch_list]
                hyps = batch_generate(model, tokenizer, queries, batch_size = gen_bs, gen_param=gen_param)
            else:
                batch,_ = next(dl_iter)
                if type(batch) == list:
                    batch = batch[0]
                hyps = generate(model, tokenizer, batch, gen_param=gen_param, task_ids=task_ids)
        else:
            #hyps = islice(infile, len(queries))
            hyps = lines[l_count: l_count + bs]
            l_count += bs 
        pbar.update(bs)
        for b, top_hyp in zip(batch_list, hyps):
            query = b["query"]
            inp = b["event"]
            tail = b["target"]
            resp = b["resp"]
            rel = b["rel"]
            qid = b["index"]
            repid = b["rep"]
            mlog.info("\n%s/%s) query: %s", step, len(test_set), query)
            mlog.info("\nhyp: %s",top_hyp)
            mlog.info("\ntail: %s",tail)
            mbp(1)
            data = {}
            if query != old_query:
                old_query = query
                iid += 1
            data["qid"] = iid
            data["tid"] = qid
            #rel_natural = relation_natural_mappings[rel]["en-postfix"]        
            #rel_natural_pure = rel_natural.replace("{ph}", "").strip()
            #top_hyp = top_hyp.replace(rel_natural_pure, "")
            blank = ""
            if "<extra_id_1>" in top_hyp:
                blank, top_hyp = top_hyp.split("<extra_id_1>")
                if not blank: blank = "EMPT"
            mlog.info("hyp: %s",top_hyp)
            resp_const = resp.split(tail)
            # constant words in target template
            affixes = []
            for rp in resp_const:
                affix = rp
                for const in resp_cost_toks:
                    affix = affix.replace(const, "")
                if affix:
                    affixes.append(affix.strip())

            for const in resp_cost_toks + affixes:
                top_hyp = top_hyp.replace(const, "")
                blank = blank.replace(const, "")
            mlog.info("hyp: %s", top_hyp)
            top_hyp = top_hyp.strip()
            if not top_hyp:
                top_hyp = "EMPT"
            data["blank"] = blank
            data["pred_text1"] = str(top_hyp)
            data["prefix"] = rel
            data["langs"] = lang
            tail = re.sub(r'<.*?>','',tail)
            tail = tail.strip()
            data["target_text"] = tail
            #if test_set.orig_df is None:
            #    data["target_text"] = tail
            input_text = re.sub(r'<.*?>','##',query)
            input_text = input_text.replace("\n", "")
            #if blank:
            #    query = query.replace("<extra_id_0>", "[" + blank + "]")
            #    query = query.replace("<extra_id_1>", ">>" + top_hyp)
            #else:
            #    query = query.replace("<extra_id_0>", ">>" + top_hyp)
            data["input_text"] = inp
            data["query"] = query 
            data["resp"] = resp
            _q = query.replace("<", "\n<", 1)
            _q = _q.replace(">", ">\n")
            data["prompt"] = _q
            rows.append(data)
            pbar.update()
            step += 1

    df = pd.DataFrame(rows)
    if test_set.orig_df is not None:
       df = test_set.orig_df.merge(df, on=['prefix','input_text'], how='inner')
    for key, info in exp_info.items():
        df[key] = info

    do_score(df, scorers, save_path)

import click
from comet.utils.find_files import *

@click.command()
@click.option(
    "--df_name",
    "-df",
    default="full",
    type=str,
    help="partial part of the filename (pattern)"
)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--scorers",
    "-sc",
    default="st",
    type=str,
    help="the name of scorers like rouge, bert or roug-bert to include both"
)
def do_score_w(df_name, path, scorers):
    files = find_files(df_name, path)
    for f in files:
        print(f)
        #r = input(f + " score?")
        if True: #r == "y":
           df = pd.read_table(f, low_memory=False)
           do_score(df, scorers, f)

import numpy as np
#import tensorflow as tf
def run_sts_benchmark(batch, embed):
  sts_encode1 = tf.nn.l2_normalize(embed(tf.constant(batch['top'].tolist())), axis=1)
  sts_encode2 = tf.nn.l2_normalize(embed(tf.constant(batch['top_pred'].tolist())), axis=1)
  cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
  clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
  scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
  """Returns the similarity scores"""
  return scores

def do_score(df, scorers, save_path, reval=False):
    #try:
    #    nltk_path = str(nltk.data.find("tokenizers/punkt"))
    #    mlog.info(f"using nltk from: {nltk_path}")
    #except LookupError:
    #    nltk.download('punkt')


# Pause the program until a remote debugger is attached
    #debugpy.wait_for_client()
    if "st" in scorers:
        embed = tf.saved_model.load("/home/pouramini/pret/sm")

    base_path = "/content/drive/MyDrive/pret"
    if not colab:
        base_path = os.path.join(home, "pret")
    local_path = f"{base_path}/paraphrase-MiniLM-L6-v2"
    if not Path(local_path).exists():
        local_path = 'sentence-transformers/paraphrase-MiniLM-L6-v2'

    bert_scorer = None
    bert_metric = None
    if "bert" in scorers:
        bert_scorer = SentenceTransformer(local_path)
        bert_metric = load_metric("bertscore")

    rouge_scorer = None
    if "rouge" in scorers:
        rouge_scorer = Rouge()

    local_path = f"{base_path}/nli-roberta-base-v2"
    if not Path(local_path).exists():
        local_path = 'sentence-transformers/nli-roberta-base-v2'
    nli_model = None
    if "nli" in scorers:
        nli_model = CrossEncoder(local_path)
    nli_counter = {}
    for l in nli_map:
        nli_counter[l] = 0
    counter = {"all":0}
    sum_match = {"all":0} 
    mean_match = {}
    sum_bert = {"all":0} 
    mean_bert = {}
    sum_rouge = {"all":0}
    mean_rouge = {}
    sum_bleu = {"all":0}
    mean_bleu = {}
    new_results = {}
    smoothie = SmoothingFunction().method4 # a function for smooth
    hyp_counter = [0]*5

    all_predictions = []
    all_golds = []
    if not reval:
        mlog.info("Preparing iterator ...")
        mlog.info("Scoring....")
    if scorers:
        rows = []
        pbar = tqdm(total=len(df), position=0, leave=True) #,dynamic_ncols=True)
        for step, row in df.iterrows():
            data = {}
            rel = row["prefix"]
            lang = row["langs"] 
            scope = rel + "_" + lang
            if not scope in sum_bert: 
                sum_bert[scope] = 0
                sum_rouge[scope] = 0
                sum_bleu[scope] = 0
                sum_match[scope] = 0
                counter[scope] = 0
            #mlog.debug("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            gen_token = gen_tokens[lang]
            #Compute embeddings
            top_hyp = str(row["pred_text1"])
            preds = [top_hyp]
            inp = row["input_text"]
            tail = re.sub(r'<extra_.*?>','',str(row["target_text"]))
            tail = tail.strip()
            tails = [tail]
            all_predictions.append(top_hyp)
            all_golds.append(tails[0])
            hi, ri = 0, 0
            hi, ri, cur_score = bert_score(bert_scorer, preds, tails)
            #summary = bert_score2(bert_metric, preds, tails)
            #cur_score = summary["bertscore_f1"]
            best_hyp = preds[hi]
            best_ref = tails[ri]
            hyp_counter[hi] += 1
            if nli_model:
                pair = (best_hyp, best_ref)
                nli_scores = nli_model.predict(pair)  
                _max  = nli_scores.argmax()
                label = nli_map[_max]
                nli_counter[label] += 1
                data["nli_group"] = label
            data["top"] = best_ref
            data["all_preds"] = "<br />".join(preds) 
            data["top_pred"] = best_hyp
            if "bert_score" in df and reval:
                df.at[step, "bert_score"] = float("{:.2f}".format(cur_score))
            else:
                data["bert_score"] = float("{:.2f}".format(cur_score))
            sum_bert[scope] += cur_score
            sum_bert["all"] += cur_score
            counter[scope] += 1
            counter["all"] += 1
            mean_bert[scope] = "{:.4f}".format(sum_bert[scope] / counter[scope])
            mean_bert["all"] = "{:.4f}".format(sum_bert["all"] / counter["all"])
            #### BLUE score
            #tokenized_rs = []
            #for r in tails:
            #    tokenized_rs.append(word_tokenize(r))
            #hypo = word_tokenize(top_hyp)
            bleu_score = 0.0
            #try:
            #    bleu_score = sentence_bleu(tokenized_rs, hypo, smoothing_function=smoothie)
            #except ValueError: # TODO ZeroDivisionError
            #    vlog.warning("math domain error in bleu, set to 0.0. generated sentence: {}".format(hypo))
            data["bleu_score"] = bleu_score 
            sum_bleu[scope] += bleu_score 
            mean_bleu[scope] = "{:.4f}".format(sum_bleu[scope] / counter[scope])
            #### Rouge score
            rouge_score = 0
            m_tails = ".".join(tails)
            m_top_hyp = top_hyp
            if rel in rel_target_omits:
                omit = rel_target_omits[rel]
                m_top_hyp = top_hyp.replace(omit, "") 
                m_tails = m_tails.replace(omit,"")
            if rouge_scorer and m_top_hyp.strip() and m_tails.strip():
                rouge_score = rouge_scorer.get_scores(m_top_hyp, m_tails, 
                                                avg=True, ignore_empty=True)
                rouge_score = rouge_score["rouge-l"]["f"]
            match_score = 0
            inp_key = inp + rel
            mean_match[scope] = "{:.4f}".format(sum_match[scope] / counter[scope])

            data["rouge_score"] = rouge_score
            sum_rouge[scope] += rouge_score
            sum_rouge["all"] += rouge_score
            mean_rouge[scope] = "{:.4f}".format(sum_rouge[scope] / counter[scope])
            mean_rouge_all = sum_rouge["all"] / counter["all"]
            mean_rouge["all"] = "{:.4f}".format(mean_rouge_all)
            pbar.set_description(f"{scope:<20} :Bert:{mean_bert[scope]:<7} | {mean_bert['all']:<7} Rouge {mean_rouge[scope]:<7}|{mean_rouge['all']:<7} ")
            step += 1
            pbar.update()
            rows.append(data)

    df2 = pd.DataFrame(rows)
    if "st" in scorers:
        sts_data = df2[["top","top_pred"]]
        preds = df2["top_pred"].to_list()
        tails = df2["top"].to_list()
        berts = [0]*len(preds)
        rouges = [0]*len(preds)
        if "bert" in scorers:
            berts = df2["bert_score"].to_list()
        if "rouge" in scorers:
            rouges = df2["rouge_score"].to_list()
        scores = []
        for batch in np.array_split(sts_data, 10):
          scores.extend(run_sts_benchmark(batch, embed))

        df2["st_score"] = ["{:.2f}".format(float(x)) for x in scores]
        res = zip(scores, berts, rouges, preds, tails)
        for s, b, r, p, t in res:
            print("{:<5.2f} {:<5.2f} {:<5.2f}{:<20} {:<20}".format(float(s),float(b),float(r),p,t))

        print("mean st score: %s", np.mean(scores))
        print("mean bert score: %s", np.mean(berts))
        print("mean rouge score: %s", np.mean(rouges))

    if not reval:
        df = pd.concat([df, df2], axis=1)

    mlog.info("Saving results %s", save_path)
    save_fname = now + "_full_results.tsv"
    if not save_path.endswith("tsv"):
        save_path = os.path.join(save_path, save_fname) 
    print("Saving results %s", save_path)
    df.to_csv(save_path, index=False, sep="\t")
    
    for metric in [mean_rouge, mean_bert, mean_match, mean_bleu]:
        s =0 
        ii = 0
        jj = 0
        for key,val in metric.items():
            metric[key] = str(val) + "--" + str(counter[key])
            s += float(val)
            ii += 1
            jj += counter[key]
        metric["AVG"] = "{:.2f}--{}".format(s/ii, jj)

    mean_bert_str = json.dumps(mean_bert, indent=2)
    mean_rouge_str = json.dumps(mean_rouge, indent=2)
    mean_bleu_str = json.dumps(mean_bleu, indent=2)
    mean_match_str = json.dumps(mean_match, indent=2)
    mlog.info("-----------------------------------------------------")
    pbar.close()
    pred_counts = df['pred_text1'].unique()
    mlog.info("Pred counts")
    vlog.info("Pred counts")
    if len(pred_counts) < 100:
        for  r in pred_counts:
            mlog.info(r)
            vlog.info(r)

    df_mean_rouge = df["rouge_score"].mean()
    for logger in [mlog, vlog, clog]:
        logger.info("Len data frame: {}".format(len(df)))
        logger.info("Rouge:{} ".format(mean_rouge_str)) 
        logger.info("DF mean Rouge Score: {}".format(df_mean_rouge))
        if "bert" in scorers:
            logger.info("BERT:{} ".format(mean_bert_str)) 
            logger.info("DF mean Bert Score: {}".format(df["bert_score"].mean()))
        #logger.info("nli_counter: {}".format(nli_counter))
        #logger.info("hyp_counter: {}".format(hyp_counter))
        logger.info("Distinct preds:{}".format(len(pred_counts)))

    return df

def write_results(exp_info, save_path, df): 
    _info = "_".join([str(x) for x in list(exp_info.values())])
    #metric_list = ["rouge", "meteor", "bertscore"]
    #summary = calc_metrics(all_predictions, all_golds, metric_list)
    summary = ""
    out = os.path.join(save_path,f"summary__{_info}.txt")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(summary)
    with open(out, "w") as f:
        print(summary, file=f)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    if not scorers:
        return


    df = df.sort_values(by=["input_text"])
    
    out = os.path.join(save_path,f"__{_info}.txt")
    def write_preds(df, out):
        handler = logging.FileHandler(out, mode="w")
        mlog.addHandler(handler)
        old_input = ""
        for i, row in df.iterrows(): 
            q = row["input_text"] 
            p = row["prefix"]
            if q != old_input:
                old_input = q
                mlog.info("\n\n")
            mlog.info("\n")
            mlog.info("{:<2} {} {:<60}:".format(i, q, p))
            preds = row["all_preds"]
            answers = row["target_text"]
            mlog.info("------------------------------------  preds for {}:".format(p))
            for pred in preds.split("<br />"):
                mlog.info("{:<60}:".format(pred))
            mlog.info("-----------------------------------  targets for {}:".format(p))
            for ans in answers.split("<br />"):
                mlog.info("{:<60}:".format(ans))



def bert_score2(metric, preds, golds):
    summary = {}
    preds = [p.strip() for p in preds]
    golds = [g.strip() for g in golds]
    res = metric.compute(predictions=preds, references=golds, lang="en", model_type="/home/pouramini/pret/paraphrase-MiniLM-L6-v2", num_layers=6)
    #res = metric.compute(predictions=preds, references=golds, lang="en", model_type="/home/pouramini/pret/t5-large", num_layers=24)
    for k, v in res.items():
        if k == "hashcode":
            continue
        summary[f"bertscore_{k}"] = round(1.0 * sum(v) / len(v), 2)
    return summary


def calc_metrics(preds, golds, metric_list):
    summary = {}
    for metric_name in metric_list:
        metric = load_metric(metric_name)
        processed_preds, processed_golds = postprocess_text(preds, golds, metric_name)

        if metric_name == "bertscore":
            res = metric.compute(predictions=preds, references=golds, lang="en", model_type="/home/pouramini/pret/paraphrase-MiniLM-L6-v2", num_layers=6)
            for k, v in res.items():
                if k == "hashcode":
                    continue
                summary[f"{metric_name}_{k}"] = round(1.0 * sum(v) / len(v), 2)

        else:
            res = metric.compute(predictions=processed_preds, references=processed_golds)
            if metric_name == "sacrebleu":
                summary[metric_name] = res["score"] * 0.01  # limit it to range of [0, 1] for unifying
            elif metric_name == "bleurt":
                summary["bleurt"] = round(1.0 * sum(res["scores"]) / len(res["scores"]), 2)
            elif metric_name == 'rouge':
                for sub_metric_name in res.keys():
                    for i, key in enumerate(['precision', 'recall', 'fmeasure']):
                        summary["{}_{}".format(sub_metric_name, key)] = res[sub_metric_name][1][i]
                    # this the the fmeasure('f-score') from the mid('mean aggregation')
            else:
                summary[metric_name] = res[metric_name]
    return summary

def acc_metric(tokenizer, all_preds, all_labels, save_path=""):
    acc = sum([int(p == l) for p, l in zip(all_preds, all_labels)]) / len(all_preds)
    
    if save_path:
        with open(os.path.join(save_path, "{}.txt".format(acc)), "w") as f:
            for p, l in zip(all_preds, all_labels):
                f.write(str(p) + "\t\t" + str(l) + "\n")
                if isinstance(p, list):
                    f.write(tokenizer.decode(p) + "\t\t" + tokenizer.decode(l) + "\n")
                f.write("\n")

    return acc


def acc_f1_metric(tokenizer, all_preds, all_labels, save_path=""):
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    acc = sum([int(p == l) for p, l in zip(all_preds, all_labels)]) / len(all_preds)

    if save_path:
        with open(os.path.join(save_path, "{}.txt".format(f1_macro)), "w") as f:
            for p, l in zip(all_preds, all_labels):
                f.write(str(p) + "\t\t" + str(l) + "\n")
                if isinstance(p, list):
                    f.write(tokenizer.decode(p) + "\t\t" + tokenizer.decode(l) + "\n")
                f.write("\n")

    return [acc, f1_macro]


if __name__ == "__main__":
    do_score_w()
