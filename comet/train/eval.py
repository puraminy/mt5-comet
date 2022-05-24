
from itertools import islice
from comet.train.common import *
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
from rouge import Rouge
#%% Aggregate instances of queries and corresponding responses
# (str)split_name -> (dict) query -> (list) response 

from datasets import load_metric
import nltk

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

def generate(model, tokenizer, batch, gen_token = "", gen_param = "greedy", at_mask=None):
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
            "max_length":60,
            "num_beams":5,
            "repetition_penalty":5.5,
            "num_return_sequences":1,
            "bad_words_ids": bad_words_ids
        }
    elif gen_param == "top_p" or gen_param == "top_k":
        gen_kwargs = {
            "max_length":60,
            "do_sample":True, 
            "top_p":0.9, 
            "top_k":10,
            "num_beams":5,
            "temperature": 1.0,
            "num_return_sequences":1, 
            "repetition_penalty":5.5,
            "bad_words_ids": bad_words_ids
        }
    batch.to(device)
    if "labels" in batch:
        gen_kwargs["labels"] = batch["labels"]
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

    #input_batch = {}
    #input_batch["input_ids"] = batch["input_ids"]
    #input_batch["attention_mask"] = batch["attention_mask"]
    #input_ids, attention_mask = trim_batch(**input_batch, pad_token_id=tokenizer.pad_token_id)
    decs = []
    if False: #gen_token != "":
        gen_token_id = tokenizer.convert_tokens_to_ids(gen_token)
        hyps = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
                decoder_start_token_id=gen_token_id)
        hyps = tokenizer.batch_decode(hyps,skip_special_tokens=False)
    else:
        hyps = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
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

def save_results(rows, fid, step, exp_info, save_path="", rewrite=False):
    pp = save_path.split("/")
    pp = "_".join(pp[-2:])
    name = fid + "_" + human_format(step) 

    if save_path:
        path = os.path.join(save_path, name + ".tsv")
    else:
        _info = "_".join([str(x) for x in list(exp_info.values())])
        path = os.path.join(resPath, name + "_" + _info + ".tsv")

    if Path(path).is_file() and rewrite:
        df = pd.read_table(path)
    else:
        df = pd.DataFrame(rows)
    if exp_info: df["val_steps"] = step
    for key, info in exp_info.items():
        df[key] = info

    mlog.info("Saving results %s", path)
    df.to_csv(path, index=False, sep="\t")
    return df
# vvvvvvvvvvvvvvv
# ################################### Evaluation #########################
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]



def evaluate(test_set, dataloader, save_path, exp_info, val_records, gen_param="greedy", scorers="rouge", batch_size="20@5", model = None, tokenizer = None, preds_file = "", set_name = "test", rewrite_info=False):  

    file_gen = "_" + Path(preds_file).stem if preds_file else ""
    ext = set_name + file_gen + "_" + scorers 
    ext = ""
    outdf_name = "full" + ext
    if rewrite_info:
        save_results([], outdf_name, len(test_set), exp_info, save_path, rewrite=True)
        return


    try:
        nltk_path = str(nltk.data.find("tokenizers/punkt"))
        mlog.info(f"using nltk from: {nltk_path}")
    except LookupError:
        nltk.download('punkt')
    base_path = "/content/drive/MyDrive/pret"
    if not colab:
        base_path = os.path.join(home, "pret")

    mlog.info("Loading models for evaluation ..")
    mlog.info("%s", save_path)

    #local_path = f"{base_path}/paraphrase-multilingual-MiniLM-L12-v2"        
    local_path = f"{base_path}/paraphrase-MiniLM-L6-v2"
    if not Path(local_path).exists():
        local_path = 'sentence-transformers/paraphrase-MiniLM-L6-v2'

    bert_scorer = None
    bert_metric = None
    if "bert" in scorers:
        bert_scorer = SentenceTransformer(local_path)
        bert_metric = load_metric("bertscore")

    rouge_score = None
    if "rouge" in scorers:
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
    resp_const_parts = ["<pad>","<extra_id_0>", "<extra_id_1>", "<extra_id_2>", "</s>", "."]
    if model is not None: model.eval()
    rows = []
    sel_rows = []
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
    ignore_special_tokens = False
    if "@" in gen_param:
        _, ist = gen_param.split("@")
        ignore_special_tokens = ist == "True"

    mlog.info("Preparing iterator ...")
    mlog.info("Scoring...")
    pbar = tqdm(total=val_records, position=0, leave=True) #,dynamic_ncols=True)
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
    test_iter = iter(test_set)
    batches = batched(list(test_iter), bs)
    if model is not None:
        dl_iter = iter(dataloader)
    iid  = 0
    old_query = ""
    all_predictions = []
    all_golds = []
    for batch_list in batches: 
        if exit_loop:
            break
        if model is not None:
            if False:
                queries = [x[0] for x in batch_list]
                hyps = batch_generate(model, tokenizer, queries, batch_size = gen_bs, gen_param=gen_param)
            else:
                batch = next(dl_iter)
                hyps = generate(model, tokenizer, batch, gen_param=gen_param)
        else:
            #hyps = islice(infile, len(queries))
            hyps = lines[l_count: l_count + bs]
            l_count += bs 
        pbar.update(bs)
        for (query, inp, tail, rel, qid, repid), top_hyp in zip(batch_list, hyps):
            tails = [tail]
            mlog.info("query: %s", query)
            mlog.info("1)  hyp: %s",top_hyp)
            data = {}
            sel_data = {}
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
            mlog.info("2)  hyp: %s",top_hyp)
            for const in resp_const_parts:
                top_hyp = top_hyp.replace(const, "")
                blank = blank.replace(const, "")
            mlog.info("3)  hyp: %s", top_hyp)
            if not top_hyp.strip():
                top_hyp = "EMPT"
            new_tails = []
            for tail in tails:
                if not tail.strip():
                    continue
                nt = tail
                #nt = nt.replace(rel_natural_pure, "")
                for const in resp_const_parts:
                    nt = nt.replace(const,"")
                new_tails.append(nt)
            tails = new_tails
            all_predictions.append(top_hyp)
            all_golds.append(tails[0])
            data["blank"] = blank
            data["pred_text1"] = top_hyp
            data["target_text"] = "<br />".join(tails)
            p_rel = rel
            if exp_info["multi"]:
                p_rel = "multi_" + rel
            data["prefix"] = p_rel
            data["langs"] = lang
            input_text = re.sub(r'<.*?>','##',query)
            input_text = input_text.replace("\n", "")
            #if blank:
            #    query = query.replace("<extra_id_0>", "[" + blank + "]")
            #    query = query.replace("<extra_id_1>", ">>" + top_hyp)
            #else:
            #    query = query.replace("<extra_id_0>", ">>" + top_hyp)
            data["input_text"] = inp
            data["query"] = query 
            if not scorers:
                if step % 10000 == 0:
                    save_results(rows, "step", step, exp_info, save_path)
                step += 1
                rows.append(data)
                continue
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
            preds = [top_hyp]
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
                vlog.info("Label:"+ label)
            data["top"] = best_ref
            data["all_preds"] = "<br />".join(preds) 
            data["top_pred"] = best_hyp
            data["bert_score"] = float("{:.2f}".format(cur_score))
            sum_bert[scope] += cur_score
            sum_bert["all"] += cur_score
            counter[scope] += 1
            counter["all"] += 1
            mean_bert[scope] = "{:.4f}".format(sum_bert[scope] / counter[scope])
            mean_bert["all"] = "{:.4f}".format(sum_bert["all"] / counter["all"])
            #tqdm.write(f"Mean score:{mean_bert}")
            vlog.info("")
            vlog.info(f"=============   {lang}  ===  {rel}   =====================")
            _q = query.replace("<", "\n<", 1)
            _q = _q.replace(">", ">\n")
            data["prompt"] = _q
            vlog.info(str(counter["all"])+ ":" + _q)
            vlog.info("'''''''''''''''''''''''''''''''''''''''''' Preds:")
            for h in preds: 
                if h == best_hyp:
                    h += " (***) " 
                vlog.info(h)
            vlog.debug('"""""""""""""""""""""""""""""""""""""""" Targets:')
            for _tail in tails:
                if _tail == best_ref:
                    _tail += "(*)" 
                vlog.debug(_tail)

            vlog.info("'''''''''''''''''''''''''''''''''''''''''''''''''''''")
            #mlog.debug(f"TOP hyp:{top_hyp}")
            #mlog.debug(f"Tails: {tails}")
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
            if rouge_scorer:
                rouge_score = rouge_scorer.get_scores(top_hyp, ".".join(tails), 
                                                avg=True, ignore_empty=True)
                rouge_score = rouge_score["rouge-l"]["f"]
            match_score = 0
            inp_key = inp + rel
            if rouge_score > 0.6 and inp_key not in sel_inps:
                match_score = 1
                sum_match[scope] += 1
                is_none = False
                if "none" in top_hyp.lower():
                    nones += 1
                    is_none = True
                if nones < 200 or not is_none:
                    sel_data["input_text"] = inp 
                    sel_data["prefix"] = rel
                    sel_data["target_text"] = tails[0]
                    sel_rows.append(sel_data)
                    sel_inps[inp_key] = True

            mean_match[scope] = "{:.4f}".format(sum_match[scope] / counter[scope])

            data["rouge_score"] = rouge_score
            sum_rouge[scope] += rouge_score
            sum_rouge["all"] += rouge_score
            mean_rouge[scope] = "{:.4f}".format(sum_rouge[scope] / counter[scope])
            mean_rouge_all = sum_rouge["all"] / counter["all"]
            if val_records > 20000 and step > int(0.5*val_records) and mean_rouge_all < 0.1:
                mlog.info("Early exit because of low score")
                exit_loop = True
                break
            mean_rouge["all"] = "{:.4f}".format(mean_rouge_all)
            #mlog.info("Bert Score:{:.4f}--{}".format(cur_score, mean_bert[scope]))
            #mlog.info("Bert Score 2:{:.4f}".format(b_score))
            #mlog.info("Rouge Score:{:.4f}--{}".format(rouge_score, mean_rouge[scope]))
            #mlog.info("Rouge Score 2:{:.4f}".format(r_score))
            #mlog.info("Match Score:{}--{}".format(match_score, mean_match[scope]))
            #vlog.info("BLEU Score:{:.4f}--{}".format(bleu_score, mean_bleu[scope]))
            vlog.info("======================================================")
            pbar.set_description(f"{scope:<20} :Bert:{mean_bert[scope]:<7} | {mean_bert['all']:<7} Rouge {mean_rouge[scope]:<7}|{mean_rouge['all']:<7} ")
            step += 1
            if step % 10000 == 0:
                save_results(rows, "step", step, exp_info, save_path)
            rows.append(data)


    # %%%%%%%%%%%%%%%%%%
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
    new_df = save_results(rows, outdf_name, step, exp_info, save_path)
    #sel_df = save_results(sel_rows, "sel" + ext , step, {}, save_path)
    if not scorers:
        return


    new_df = new_df.sort_values(by=["input_text"])
    
    out = os.path.join(save_path,f"__{_info}.txt")
    def write_preds(new_df, out):
        handler = logging.FileHandler(out, mode="w")
        mlog.addHandler(handler)
        old_input = ""
        for i, row in new_df.iterrows(): 
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
    with open(out, "a") as f: 
        print("{:<40}:".format(mean_rouge_str), file = f)
        print("{:<40}:".format(mean_bert_str), file = f)
        print("{:<40}:".format(mean_bleu_str), file = f)
        print("{:<40}:".format(mean_match_str), file = f)
    mlog.info("-----------------------------------------------------")
    pbar.close()
    pred_counts = new_df['pred_text1'].unique()
    mlog.info("Pred counts")
    vlog.info("Pred counts")
    if len(pred_counts) < 100:
        for  r in pred_counts:
            mlog.info(r)
            vlog.info(r)

    df_mean_rouge = new_df["rouge_score"].mean()
    for logger in [mlog, vlog, clog]:
        logger.info("Len data frame: {}".format(len(new_df)))
        logger.info("Rouge:{} ".format(mean_rouge_str)) 
        logger.info("DF mean Rouge Score: {}".format(df_mean_rouge))
        if "bert" in scorers:
            logger.info("BERT:{} ".format(mean_bert_str)) 
            logger.info("DF mean Bert Score: {}".format(new_df["bert_score"].mean()))
        #logger.info("nli_counter: {}".format(nli_counter))
        #logger.info("hyp_counter: {}".format(hyp_counter))
        logger.info("Distinct preds:{}".format(len(pred_counts)))


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
