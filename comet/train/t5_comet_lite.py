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
@click.option(
    "--model_id",
    default="t5-base",
    type=str,
    help=""
)
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
    "--iterations",
    "-i",
    default=2000,
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
    "--base",
    "-base",
    default="validation",
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

def main(model_id, path, input_text, target_text, from_dir, iterations, val_set, 
         num_generations, is_flax, base, lang):
    #%% some hyper-parameters
    #underlying_model_name = "logs/atomic-mt5/last"
    if from_dir:
        underlying_model_name = path
    else:
        underlying_model_name = f"/home/pouramini/pret/{model_id}"
    if not Path(underlying_model_name).exists():
        underlying_model_name = model_id
       
    learning_rate = 6.25e-5
    cycle = 1000 #500
    warm_up_steps = 0.002*iterations
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
    log_dir = 'logs/' if not base else base + "/logs/"
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    model_name = f"{learning_rate}_{cycle}_{iterations}"
    serialization_dir = os.path.join(log_dir,model_id)
    ii = 1
    while Path(serialization_dir).exists():
        ans = input("The output directory already exists, do you want to load the model from it? (y/n)")
        if ans == "y":
            underlying_model_name = serialization_dir
        serialization_dir = os.path.join(log_dir,model_name, "_"+str(ii))
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
    lang_tokens = [gen_token_en, gen_token_fa]
    targets = ["target_text", "target_text_fa"]
    inputs = ["input_text", "input_text_fa"]
    
    #%% Aggregate instances of queries and corresponding responses
    # (str)split_name -> (dict) query -> (list) response 
    print("building query responses")
    atomic_query_responses = {}
    for split_name,split_data in atomic_dataset.items():
        atomic_query_responses[split_name] = {}
        split_data[target_text] = split_data[target_text].astype(str)
        for index, d in split_data.iterrows():
            rel = d["prefix"]
            if len(d[target_text])>0: 
                for inp in inputs:
                    rel_token = atomic_relation_mappings[rel]
                    event = d[inp]
                    query = f"{rel_token}: {event}"
                    if query not in atomic_query_responses[split_name]:
                        atomic_query_responses[split_name][query] = []                
                    for lang_token, col in zip(lang_tokens, targets):
                        resp = lang_token + d[col]
                        atomic_query_responses[split_name][query].append(resp)
                #didn't convert ___ to <blank>
                #didn't normalize to lowercase

    #flatten
    print("building flattened pairs")
    atomic_flattened = {}
    for split_name,queries_responses in atomic_query_responses.items():
        atomic_flattened[split_name] = []
        for query,responses in queries_responses.items():
            for response in responses:
                atomic_flattened[split_name].append((query,response))
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
            lang_tokens
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
    sw = SummaryWriter(serialization_dir)
    tokenizer.save_pretrained(serialization_dir)
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
                    model.save_pretrained(serialization_dir)
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
