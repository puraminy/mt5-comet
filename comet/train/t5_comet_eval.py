from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
import pandas as pd
# ################################### Evaluation #########################
def eval(model, data_df, val_set, num_generations, inter, save_path):  
    mname = "localpath"        
    if not Path(mname).exists():
        mname = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    scorer_model = SentenceTransformer(mname)
    mname = "localpath"
    if not Path(mname).exists():
        mname = 'sentence-transformers/nli-roberta-base'
    nli_model = CrossEncoder(mname)
    results = []
    labels_count = {}
    for l in nli_map:
        labels_count[l] = 0
    df = data_df[val_set]
    inp = split_col[val_set]["input_text"]
    target = split_col[val_set]["targets"][0]
    print("Len Final Df:", len(df))
    df[target] = df[target].astype(str)
    #df = df.groupby(['prefix','input_text'],as_index=False)[target].agg({"target_text":'<br />'.join})
    resp_const_parts = re.split("{.*}", anstemp)
    resp_const_parts += ["<extra_id_0>", "<extra_id_1>"]
    print("Scoring...")
    df["pred1_score"] = 0.0
    df["pred_text1"] = ""
    df["nli_group"] = ""
    df["top"] = ""
    df.sort_values(by=["prefix", "input_text"], inplace=True)

    model.eval()
    sum_score = 0 
    total = num_generations
    #if num_generations == 0:
    total = min(num_generations, len(df))
    ii = 0
    old_input = ""
    max_score = 0
    jj =0
    pbar = tqdm(total = total)
    exit_loop = False
    for idx,row in df.iterrows():
        if num_generations>0 and ii >= num_generations:
            print("break at ", ii)
            break
        jj += 1
        rel = row["prefix"]

        encoder_rel_tokens = encoder_relation_mappings[rel]
        decoder_rel_tokens = decoder_relation_mappings[rel]
        #query = f"{rel_tokens} {row['event']}" #Change this line to modify the encoder input
        event = row[inp]
        if inter: #interactive mode
            while True:
                try:
                    user_event = input("Enter event or skip (Enter), c)ontinue, e)xit:")
                    if user_event == "e":
                        exit_loop = True
                    if user_event == "c":
                        inter = False
                    if user_event != "":
                        event = user_event
                    break
                except KeyboardInterrupt:
                    break
                except:
                    print("Error occured, please try again or hit e for exit")
                    continue
        if exit_loop:
            break

        query = qtemp.format(event=event, rel=encoder_rel_tokens, ph=placeholder_token) 
        gen_token = gen_token_fa if target=="target_text_fa" else gen_token_en
        gen_token_id = tokenizer.convert_tokens_to_ids(gen_token)
        if "{gen}" in anstemp:
            hyps = tokenizer.batch_decode(
                            model.generate(**tokenizer(query, return_tensors='pt').to(device=device),
                                           decoder_start_token_id=gen_token_id,**generation_params),
                            skip_special_tokens=True
                        )
        else:
            hyps = tokenizer.batch_decode(
                            model.generate(**tokenizer(query, return_tensors='pt').to(device=device),
                                           **generation_params),
                            skip_special_tokens=True
                        )

        query = re.sub(r'<.*?>','',query)
        #tails = row[target].split("<br />")
        tails = [row[target]] if event == row[inp] else "NA"
        #tails = [x[1] for x in responses]
        new_hyps = []
        for hyp in hyps:
            if hyp == "":
                hyp = "."
            for const in resp_const_parts:
                hyp = hyp.replace(const, "")
            new_hyps.append(hyp)

        hyps = new_hyps

        sents1 = tails
        sents2 = hyps

        #Compute embeddings
        embeddings1 = scorer_model.encode(sents1, device=device, convert_to_tensor=True)
        embeddings2 = scorer_model.encode(sents2, device=device, convert_to_tensor=True)

        #Compute cosine-similarities for each sentence with each other sentence
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

        #Find the pairs with the highest cosine similarity scores
        pairs = []
        rows = cosine_scores.shape[0]
        cols = cosine_scores.shape[1]
        for i in range(rows):
            for j in range(cols):
                pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
            #print({'index': [i, j], 'score': cosine_scores[i][j]})

        #Sort scores in decreasing order
        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

        top = pairs[0]
        pred_text = str(sents2[top["index"][1]])
        closest = str(sents1[top["index"][0]])
        pair = (closest, pred_text)
        nli_scores = nli_model.predict(pair)
        _max  = nli_scores.argmax()
        label = nli_map[_max]
        labels_count[label] += 1
        #cond = (df['prefix'] == rel) & (df[inp] == query)
        df.at[idx, "nli_group"] = label
        df.at[idx, "top"] = closest

        df.at[idx, "pred_text1"] = pred_text
        df.at[idx, "all_preds"] = "<br />".join(hyps) 
        cur_score = top["score"]
        df.at[idx, "pred1_score"] = float("{:.2f}".format(cur_score))
        print("")
        if row["input_text"] != old_input:
            old_input = row["input_text"]
            sum_score += (max_score if max_score > 0 else cur_score)
            max_score = cur_score
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            ii += 1
            pbar.update(1)
        elif cur_score > max_score:
            max_score = cur_score
            print("======================================================")

        mean_score = "{:.4f}".format(sum_score / ii)
        #tqdm.write(f"Mean score:{mean_score}")
        if slow > 0:
            time.sleep(slow)
        print(ii, "::",query)
        print("Prediction:", pred_text)
        print("Closest tail:", closest)
        print("Label:", label)
        print("------------------------------------------------------")
        pbar.set_description(f"Mean score:{mean_score} cur score {cur_score:.2f} max score:{max_score:.2f}")

        results.append({
            "head":query,
            "gens":hyps,
            "tails":tails,
        })
        # %%%%%%%%%%%%%%%%%%
    df = df[df["pred1_score"] > 0]
    pbar.close()
    genfile = os.path.join(save_path,f"{val_set}_gen.json")
    with open(genfile,'w') as f:
        json.dump(results,f,ensure_ascii=False,indent=2)

    from comet.evaluation.rouge.rouge import Rouge
    scorer = Rouge()
    resfile = os.path.join(save_path,f"{val_set}_results.json")
    refs = {r['head']:r['tails'] for r in results}
    hyps = {r['head']:r['gens'] for r in results}
    score,_ = scorer.compute_score(hyps, refs)
    res_out = open("results", "w" if clear else "a")
    print(f"###################################### {model_id} #################")
    print(f"###################################### {model_id} #################", file=res_out)
    print(model_name, file=res_out)
    print("val_set:", val_set, file=res_out)
    print("train file:", train_df_path, file=res_out)
    print("traing samples:", inp_samples, " unique inputs, ",  iterations, " total samples",file=res_out)
    print("ignore blanks:", ignore_blanks, file=res_out)
    print("treshold_score:",tresh_score, file=res_out)
    print("nli_group:",nli_group, file=res_out)

    print("# *****************************************************************", file=res_out)
    print("Rouge:", score)
    print("Rouge:", score, file = res_out)
    # %% save dataframe %
    score_col = "pred1_score"
    col2 = "target_text" 
    out1 = "data/" + val_set + "_" + model_name  + ".tsv" 
    df.to_csv(out1, sep="\t", index=False)

    df = df.sort_values(score_col, ascending=False).\
      drop_duplicates(['prefix','input_text']).\
        rename(columns={col2:'top'}).\
          merge(df.groupby(['prefix','input_text'],as_index=False)[col2].agg('<br />'.join))
    print("Bert Score:", df["pred1_score"].mean())
    print("Bert Score:", df["pred1_score"].mean(), file=res_out)
    print("labels_count:", labels_count)
    print("labels_count:", labels_count, file=res_out)

    pred_counts = df['pred_text1'].unique()

    print("Distinct preds:", len(pred_counts))
    print("Distinct preds:", len(pred_counts), file=res_out)
    df_path = (save_path if save_path.startswith("/") else path + "/" + save_path)  
    out = df_path + "/scored_" + model_name  + ".tsv" 
    print(out)
    print(len(df))

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("'qtemp':", qtemp)
    print("'qtemp':", qtemp, file=res_out)
    print("'anstemp':", anstemp)
    print("'anstemp':", anstemp, file=res_out)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    df.to_csv(out, sep="\t", index=False)
    with open("/home/pouramini/dflist", "w" if clear else "a") as dflist:
        print(f"{model_name}={out}", file=dflist)
    res_out.close()

