
from comet.train.common import *
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
from rouge import Rouge
#%% Aggregate instances of queries and corresponding responses
# (str)split_name -> (dict) query -> (list) response 

results = {}
resFile = os.path.join(resPath, "results.json")
if Path(resFile).exists():
    with open(resFile, "r") as f:
        mlog.info("Reading stored results...")
        results = json.load(f)

results_full = {}
resFile = os.path.join(resPath, "results_full.json")
if Path(resFile).exists():
    with open(resFile, "r") as f:
        mlog.info("Reading stored results full...")
        results_full = json.load(f)

def reset_results():
    global results, results_full
    with open(os.path.join(resPath, f"results_{now}.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(resPath, f"results_full_{now}.json"), "w") as f:
        json.dump(results_full, f, indent=2)
    results = {}
    results_full = {}

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

# ggggggggg
def gen_resp(model, tokenizer, query, gen_token = "", gen_param = "greedy", at_mask=None):
    skip_special = "True"
    #verb = get_verb(query)
    #vlog.info("Ignoring verb %s", verb)
    bad_words_ids = None
    #if verb:
    #    bad_words_ids = tokenizer(verb).input_ids
    if "@" in gen_param:
        gen_param, skip_special = gen_param.split("@")
    if gen_param == "greedy":
        generation_params = {
            "max_length":260,
            "num_beams":4,
            "repetition_penalty":5.5,
            "num_return_sequences":3,
            "bad_words_ids": bad_words_ids
        }
    elif gen_param == "top_p":
        generation_params = {
            "max_length":260,
            "do_sample":True, 
            "top_p":0.9, 
            "top_k":10,
            "num_beams":5,
            "temperature": 1.0,
            "num_return_sequences":3, 
            "repetition_penalty":3.5,
            "bad_words_ids": bad_words_ids
        }
    inputs = tokenizer(query,return_tensors='pt').to(device=device)
    if False: #gen_token != "":
        gen_token_id = tokenizer.convert_tokens_to_ids(gen_token)
        hyps = model.generate(**inputs,**generation_params,
                decoder_start_token_id=gen_token_id)
        hyps = tokenizer.batch_decode(hyps,skip_special_tokens=True)
    else:
        hyps = model.generate(**inputs,**generation_params)
        hyps = tokenizer.batch_decode(hyps,skip_special_tokens=skip_special == "True")
    return hyps

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
def eval(model, tokenizer, val_data, interactive, save_path, results_info, val_records, gen_param="greedy", at_mask = None):  

    try:
        nltk_path = str(nltk.data.find("tokenizers/punkt"))
        mlog.info(f"using nltk from: {nltk_path}")
    except LookupError:
        nltk.download('punkt')
    base_path = "/content/drive/MyDrive/pret"
    if not colab:
        base_path = os.path.join(home, "pret", "mm")

    mlog.info("Loading models for evaluation ..")

    local_path = f"{base_path}/paraphrase-multilingual-MiniLM-L12-v2"        
    if not Path(local_path).exists():
        local_path = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    if "ahmad" in home:
        bert_scorer = None
    else:
        bert_scorer = SentenceTransformer(local_path)
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
    resp_const_parts = ["<extra_id_0>", "<extra_id_1>", "<extra_id_2>", "</s>", "."]
    mlog.info("Scoring...")
    model.eval()
    pbar = tqdm(total = val_records)
    rows = []
    counter = {"all":0}
    sum_match = {} 
    mean_match = {}
    sum_bert = {} 
    mean_bert = {}
    sum_rouge = {}
    mean_rouge = {}
    sum_bleu = {}
    mean_bleu = {}
    smoothie = SmoothingFunction().method4 # a function for smooth
    hyp_counter = [0]*5
    ignore_special_tokens = False
    if "@" in gen_param:
        _, ist = gen_param.split("@")
        ignore_special_tokens = ist == "True"

    for rel in val_data.keys():
        vlog.info(f"%%%%%%%%%%%%%%%%%%%%%%%%% {rel} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        for lang in val_data[rel].keys():
            vlog.info(f"%%%%%%%%%%%%%%%%%%%%%%%%%% { lang } %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            for queries in val_data[rel][lang]:
                for index_query, tails in queries.items():
                    qid = index_query[0]
                    query = index_query[1]
                    data = {}
                    scope = rel + "_" + lang
                    if not scope in sum_bert: 
                        sum_bert[scope] = 0
                        sum_rouge[scope] = 0
                        sum_bleu[scope] = 0
                        sum_match[scope] = 0
                        counter[scope] = 0
                    vlog.debug("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                    if interactive: #interactive mode
                        query = get_input("Enter an even or Enter) skip, c) continue, e) exit.")
                        resp = "NA"
                        if query == "e":
                            return data_split, flat_data, kk
                        if query == "c":
                            interactive = False
                    gen_token = gen_tokens[lang]
                    hyps = gen_resp(model, tokenizer, query, gen_token, gen_param, at_mask)
                    input_text = re.sub(r'<.*?>','##',query)
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
                    data["qid"] = qid
                    data["pred_text1"] = top_hyp
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
                    data["input_text"] = input_text 
                    data["all_preds"] = "<br />".join(hyps) 
                    data["target_text"] = "<br />".join(tails)
                    data["prefix"] = rel
                    data["langs"] = lang
                    data["top_pred"] = best_hyp
                    data["bert_score"] = float("{:.2f}".format(cur_score))
                    sum_bert[scope] += cur_score
                    counter[scope] += 1
                    counter["all"] += 1
                    mean_bert[scope] = "{:.4f}".format(sum_bert[scope] / counter[scope])
                    #tqdm.write(f"Mean score:{mean_bert}")
                    vlog.info("")
                    vlog.info(f"=============   {lang}  ===  {rel}   =====================")
                    _q = query.replace("<", "\n<", 1)
                    _q = _q.replace(">", ">\n")
                    data["prompt"] = _q
                    vlog.info(str(counter["all"])+ ":" + _q)
                    vlog.info("'''''''''''''''''''''''''''''''''''''''''' Preds:")
                    for h in hyps: 
                        if h == best_hyp:
                            h += " (***) " 
                        vlog.info(h)
                    vlog.debug('"""""""""""""""""""""""""""""""""""""""" Targets:')
                    for _tail in tails:
                        if _tail == best_ref:
                            _tail += "(*)" 
                        vlog.debug(_tail)

                    vlog.info("'''''''''''''''''''''''''''''''''''''''''''''''''''''")
                    mlog.debug(f"TOP hyp:{top_hyp}")
                    mlog.debug(f"Tails: {tails}")
                    #### BLUE score
                    tokenized_rs = []
                    for r in tails:
                        tokenized_rs.append(word_tokenize(r))
                    hypo = word_tokenize(top_hyp)
                    try:
                        bleu_score = sentence_bleu(tokenized_rs, hypo, smoothing_function=smoothie)
                    except ValueError: # TODO ZeroDivisionError
                        vlog.warning("math domain error in bleu, set to 0.0. generated sentence: {}".format(hypo))
                        bleu_score = 0.0
                    data["bleu_score"] = bleu_score 
                    sum_bleu[scope] += bleu_score 
                    mean_bleu[scope] = "{:.4f}".format(sum_bleu[scope] / counter[scope])
                    #### Rouge score
                    rouge_score = rouge_scorer.get_scores(top_hyp, ".".join(tails), 
                                                        avg=True, ignore_empty=True)
                    rouge_score = rouge_score["rouge-l"]["f"]
                    match_score = 0
                    if rouge_score > 0.9:
                        match_score = 1
                        sum_match[scope] += 1
                    mean_match[scope] = "{:.4f}".format(sum_match[scope] / counter[scope])

                    data["rouge_score"] = rouge_score
                    sum_rouge[scope] += rouge_score
                    mean_rouge[scope] = "{:.4f}".format(sum_rouge[scope] / counter[scope])
                    vlog.info("Bert Score:{:.4f}--{}".format(cur_score, mean_bert[scope]))
                    vlog.info("Rouge Score:{:.4f}--{}".format(rouge_score, mean_rouge[scope]))
                    vlog.info("Match Score:{}--{}".format(match_score, mean_match[scope]))
                    vlog.info("BLEU Score:{:.4f}--{}".format(bleu_score, mean_bleu[scope]))
                    vlog.info("======================================================")
                    pbar.set_description(f"{scope} :Bert:{mean_bert[scope]} Rouge {mean_rouge[scope]} Bleu {mean_bleu[scope]} Match {mean_match[scope]}")
                    pbar.update(1)
                    dictPath(results_info, results_full, data, sep="_")
                    rows.append(data)

    # %%%%%%%%%%%%%%%%%%
    out = os.path.join(logPath,f"__{results_info}.txt")

    handler = logging.FileHandler(out, mode="w")
    mlog.addHandler(handler)

    new_df = pd.DataFrame(rows)
    new_df = new_df.sort_values(by=["input_text"])
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
    out1 = os.path.join(save_path,f"scored_{results_info}.tsv")
    out2 = os.path.join(resPath,f"scored_{results_info}.tsv")
    out3 = os.path.join(logPath,f"scored_{results_info}.tsv")

    new_df.to_csv(out1, sep="\t", index=False)
    new_df.to_csv(out2, sep="\t", index=False)
    if colab:
       new_df.to_csv(out3, sep="\t", index=False)
    pred_counts = new_df['pred_text1'].unique()

    res = {}
    res["rouge"] = mean_rouge
    res["bert"] = mean_bert
    #res["bleu"] = mean_bleu
    #res["match"] = mean_match
    res["distinct"] ="{} {:.2f}".format(len(pred_counts), len(pred_counts)/len(new_df))
    res["hyps"] = hyp_counter

    dictPath(results_info, results, res, sep="_")
    with open(os.path.join(resPath, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(logPath, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


    with open(os.path.join(resPath, "results_full.json"), "w") as f:
        json.dump(results_full, f, indent=2)
    with open(os.path.join(logPath, "results_full.json"), "w") as f:
        json.dump(results_full, f, indent=2)

    for logger in [mlog, vlog, clog]:
        logger.info("Len data frame: {}".format(len(new_df)))
        logger.info("Rouge:{} Match: {} BERT: {} BLEU: {}".format(mean_rouge_str, 
            mean_match_str, mean_bert_str, mean_bleu_str))
        logger.info("DF mean Bert Score: {}".format(new_df["bert_score"].mean()))
        logger.info("DF mean Rouge Score: {}".format(new_df["rouge_score"].mean()))
        logger.info("nli_counter: {}".format(nli_counter))
        logger.info("hyp_counter: {}".format(hyp_counter))
        logger.info("Distinct preds:{}".format(len(pred_counts)))

