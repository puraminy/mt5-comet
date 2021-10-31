#%% load libraries

from sentence_transformers import SentenceTransformer, util
from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast, 
    MT5ForConditionalGeneration, MT5TokenizerFast, AdamW, AddedToken,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
import torch
import re
import json
from torch.utils.tensorboard import SummaryWriter
import os,time
import argparse
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import click
from tqdm import tqdm
@click.command()
@click.argument("model_id", type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--input_text",
    default="input_text",
    type=str,
    help=""
)
@click.option(
    "--target_text",
    default="target_text",
    type=str,
    help=""
)
@click.option(
    "--from_dir",
    "-",
    is_flag=True,
    help=""
)
@click.option(
    "--num_samples",
    "-n",
    default=100,
    type=int,
    help=""
)
@click.option(
    "--val_set",
    default="validation",
    type=str,
    help=""
)
@click.option(
    "--num_generations",
    "-g",
    default=0,
    type=int,
    help=""
)
@click.option(
    "--is_flax",
    "-if",
    is_flag=True,
    help=""
)
@click.option(
    "--load_path",
    "-load",
    default="/home/pouramini/pret",
    type=str,
    help=""
)
@click.option(
    "--overwrite",
    "-o",
    is_flag=True,
    help=""
)
@click.option(
    "--save_path",
    "-save",
    default="logs",
    type=str,
    help=""
)
@click.option(
    "--lang",
    "-lang",
    default="lang",
    type=str,
    help=""
)
@click.option(
    "--qtemp",
    "-qt",
    default="{rel_token}: {event} {gen} {ph}",
    type=str,
    help="template for query"
)
@click.option(
    "--anstemp",
    "-at",
    default="{ph} {response}",
    type=str,
    help="tempate for response"
)
@click.option(
    "--pred_tresh",
    "-pred_tresh",
    default=0,
    type=int,
    help="Minimum prediction score to be selected"
)
@click.option(
    "--ignore_blanks",
    "-ib",
    is_flag=True,
    help=""
)
@click.option(
    "--nli_group",
    "-nli",
    default="all",
    type=str,
    help="nli group"
)
@click.option(
    "--learning_rate",
    "-lr",
    default=6.25e-5,
    type=float,
    help="learning rate"
)
def main(model_id, path, input_text, target_text, from_dir, num_samples, val_set, 
         num_generations, is_flax, load_path, overwrite, save_path, lang, qtemp, anstemp, pred_tresh, ignore_blanks, nli_group, learning_rate):
    #%% some hyper-parameters
    #underlying_model_name = "logs/atomic-mt5/last"
    if from_dir:
        underlying_model_name = path
    elif Path(load_path).exists():
        underlying_model_name = f"{load_path}/{model_id}"
        if not Path(underlying_model_name).exists():
            underlying_model_name = model_id        
    else:
        underlying_model_name = model_id
        
    cycle = 1000 #500
    weight_decay = 0.01
    batch_size = 1
    shuffle = False
    shuffle_evaluation=False
    validation_size = 100 #10000
    validation_num_generation = 10
    generation_params = {
        "max_length":80,
        "early_stopping":True
    }
    device = 'cuda'
    log_dir = save_path
    save_path = os.path.join(log_dir, model_id)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    model_name = f"{learning_rate}_{cycle}_{num_samples}"
    print("SAVE Path:", save_path)
    ii = 1
    while not overwrite and Path(save_path).exists():
        ans = input("The output directory already exists, do you want to load the model from it? (y/n)")
        if ans == "y":
            underlying_model_name = save_path
            overwrite = True
        save_path = os.path.join(log_dir,model_id, "_"+str(ii))
        ii += 1


    #%% load atomic data
    import pandas as pd
    atomic_dataset = {}
    train_path= "atomic/xIntent_en_fa_train_no_dups.tsv"
    val_path= "atomic/xIntent_en_fa_validation_no_dups.tsv"
    atomic_dataset["train"] = pd.read_table(train_path)
    atomic_dataset["validation"] = pd.read_table(val_path)

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
    gen_token_en = "<gen_en>"
    gen_token_fa = "<gen_fa>"
    gen_tokens = {"target_text":gen_token_en, 
                  "target_text_fa":gen_token_fa,
                  "pred_text1":gen_token_en,
                  "pred_text_fa":gen_token_fa,
                  "all_preds":gen_token_en,
                  "all_preds_fa":gen_token_fa}
                  
    targets = ["target_text", "target_text_fa", "pred_text1", "all_preds", "pred_text_fa","all_preds_fa"]
    inputs = ["input_text", "input_text_fa"]

    placeholder_token = "<extra_id_0>"
    end_token = "<extar_id_1>"

    def format_temp(template, rel, event, gen_token, resp):
        rel_token = atomic_relation_mappings[rel]        
        return template.format(event=event, 
                             response=resp,
                             rel=rel, 
                             rel_token=rel_token,
                             gen=gen_token,
                             ph=placeholder_token,                                                                       end=end_token)

    #%% Aggregate instances of queries and corresponding responses
    # (str)split_name -> (dict) query -> (list) response 
    # mmmmmmmmmmmmmm
    def my_load_dataset(split_df, inputs, targets, 
                        num_samples=0, 
                        ignore_blanks=False,
                        pred_tresh=0,
                        nli_group="all"):
        print("building query responses")
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
            print("*** Filtered based on pred1 score higher than ", pred_tresh)
        if nli_group != "all" and "nli_group" in split_df:
            split_df = split_df[split_df["nli_group"] == nli_group]
            print("*** Filtered based on nli_group ", nli_group)

        split_df[target_text] = split_df[target_text].astype(str)
        jj = 0
        ii = 0
        for index, d in split_df.iterrows():
            rel = d["prefix"]
            for inp in inputs:
                for targ_col in targets:
                    if not targ_col in d or len(d[targ_col]) <= 1:
                        continue
                    rel_token = atomic_relation_mappings[rel]
                    event = d[inp]
                    resp = d[targ_col]
                    gen_token = gen_tokens[targ_col]
                    query = format_temp(qtemp, rel, event, gen_token, resp) 
                    resp = format_temp(anstemp, rel, event, gen_token, resp)
                    if query not in data_split:
                        jj+=1
                        if jj >= num_samples:
                            return data_split
                        data_split[query] = []                
                        if ii < 3:
                            print("Q:",query)
                            print("R:",resp)
                        ii+=1
                    data_split[query].append(resp)
                #didn't convert ___ to <blank>
                #didn't normalize to lowercase
        return data_split

    atomic_query_responses = {}
    for split_name,split_df in atomic_dataset.items():
        atomic_query_responses[split_name] = my_load_dataset(split_df, inputs, targets,
                                                            num_samples, 
                                                            ignore_blanks, 
                                                            pred_tresh, nli_group)
    #flatten
    def flatten(atomic_query_responses):
        atomic_flattened = {}
        kk = 0
        for split_name,queries_responses in atomic_query_responses.items():
            print("building flattened pairs for ", split_name)
            atomic_flattened[split_name] = []
            for query,responses in queries_responses.items():
                for response in responses:
                    atomic_flattened[split_name].append((query,response))
                    kk +=1
        return atomic_flattened,kk

    atomic_flattened, iterations = flatten(atomic_query_responses)
    print("Iterations:", iterations)
    warm_up_steps = 0.002*iterations
    #%% tokenizer & model
    if "mt5" in model_id:
        tokenizer = MT5TokenizerFast.from_pretrained(underlying_model_name)
        model = MT5ForConditionalGeneration.from_pretrained(underlying_model_name)
    elif is_flax:
        tokenizer = AutoTokenizer.from_pretrained(underlying_model_name)
        model = T5ForConditionalGeneration.from_pretrained(underlying_model_name, from_flax=True) 
    else:
        tokenizer = AutoTokenizer.from_pretrained(underlying_model_name)
        model = T5ForConditionalGeneration.from_pretrained(underlying_model_name) 

    # add new tokens
    # added_tokens = list(atomic_relation_mappings.values()) + [gen_token]
    added_tokens = [ 
        AddedToken(token,lstrip=True,
            rstrip=False)
        for token in 
            list(atomic_relation_mappings.values())+
            list(gen_tokens.values())
    ]
    tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
    model.resize_token_embeddings(len(tokenizer))
    #%% Prepare training data

    def collate_fn_for_flattened(batch):
        queries,responses = zip(*batch)
        new_batch = tokenizer(list(queries),return_tensors='pt',padding='longest')
        with tokenizer.as_target_tokenizer():
            outputs = tokenizer(list(responses),return_tensors='pt',padding='longest')
            labels = outputs['input_ids']
            labels[labels==tokenizer.pad_token_id] = -100
            new_batch['labels']=labels
        return new_batch

    # def collate_fn_for_generation(batch):
    #     queries,references = zip(*batch)
    #     new_batch = tokenizer(queries,return_tensors='pt',padding='longest')
    #     return new_batch,references
    #%% build dataloader
    train_dataloader = torch.utils.data.DataLoader(atomic_flattened['train'],
        batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn_for_flattened)
    dev_dataloader = torch.utils.data.DataLoader(atomic_flattened['validation'],
        batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn_for_flattened)
    # %% prepare for training
    sw = SummaryWriter(save_path, flush_secs=1)
    tokenizer.save_pretrained(save_path)
    model = model.to(device=device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate,eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,warm_up_steps,iterations)
    step = 0
    best_dev_loss = 1e10

    #%%
    train_iter = iter(train_dataloader)
    pbar = tqdm(total=iterations) #,dynamic_ncols=True)
    while step <= iterations:
        if (step % cycle == 0 and step > 0): #validation
            with torch.no_grad():
                model.eval()
                pbar.set_description('validating...')
                dev_allset_micro_loss = 0.
                dev_token_loss = 0.
                dev_token_count = 0
                dev_sample_loss = 0. #avg on sample
                dev_sample_count = 0
                for batch in tqdm(dev_dataloader,desc=f'validating...',leave=False):
                    if dev_sample_count>=validation_size:
                        break
                    batch = {k:v.to(device=device) for k,v in batch.items()}
                    result = model(**batch)
                    loss = torch.nn.functional.cross_entropy(
                        result['logits'].reshape(-1,result['logits'].size(2)),
                        batch['labels'].reshape(-1,),
                        reduction='none'
                    ).reshape(result['logits'].size(0),-1)
                    labels_mask = (batch['labels'] != -100) 
                    dev_token_loss += loss.sum().item()
                    dev_token_count += labels_mask.sum().item()
                    dev_sample_loss += (loss.sum(dim=-1)/labels_mask.sum(dim=-1)).sum().item()
                    dev_sample_count += result['logits'].size(0)
                    del result
                    del loss
                    del labels_mask
                dev_micro_avg_loss = dev_token_loss/dev_token_count
                dev_macro_avg_loss = dev_sample_loss/dev_sample_count
                sw.add_scalar('dev/micro_avg_loss',dev_micro_avg_loss,step)
                sw.add_scalar('dev/macro_avg_loss',dev_macro_avg_loss,step)
                if dev_micro_avg_loss < best_dev_loss:
                    best_dev_loss = dev_micro_avg_loss
                    model.save_pretrained(save_path)
                    with open(save_path + "/best_model.txt", "a") as f:
                        print("step:", step, file=f)
                        print("best dev loss:", best_dev_loss, file=f)

                generation_results = \
                "|Queries|Generation Results|\n"\
                "|-|-|\n"
                for i,key in enumerate(atomic_query_responses['validation']):
                    if i==validation_num_generation:
                        break
                    results = tokenizer.batch_decode(
                        model.generate(**tokenizer(key,return_tensors='pt').to(device=device),**generation_params),
                        skip_special_tokens=True
                    )
                    generation_results+=f"|`{key}`|`{str(results)}`|\n"
                sw.add_text('dev/generation_samples',generation_results,step)
        pbar.set_description('training...')
        pbar.update()
        model.train()
        optimizer.zero_grad()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
        batch = {k:v.to(device=device) for k,v in batch.items()}
        result = model(**batch)
        loss = result['loss']
        loss.backward()
        optimizer.step()
        scheduler.step()
        step+=1
        sw.add_scalar('train/loss',loss.item(),global_step=step)
        del result
        del loss
    pbar.close()
    sw.close()
    # %%
    # %%
if __name__ == "__main__":
    main()
