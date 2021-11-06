# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
# ---

# %% load libraries
from comet.train.t5_comet_eval import *

from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast, AdamW, AddedToken,
    MT5ForConditionalGeneration, MT5TokenizerFast,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from comet.transformers_ptuning import PTuningWrapper
import torch
import csv 
import re
import json
from torch.utils.tensorboard import SummaryWriter
import os,time
import argparse
from tqdm.auto import tqdm
from comet.transformers_ptuning.ptuning_wrapper import LSTMEmbeddingPromptEncoder, EmbeddingPromptEncoder
# %% argparse
from pathlib import Path
import pandas as pd
import click
from tqdm import tqdm
@click.command()
@click.argument("model_id", type=str)
@click.argument("exp_id", type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--inp_samples",
    "-i",
    default=500,
    type=int,
    help=""
)
@click.option(
    "--cycle",
    "-c",
    default=1000,
    type=int,
    help=""
)
@click.option(
    "--frozen",
    "-f",
    is_flag=True,
    help="keep model frozen or not"
)
@click.option(
    "--sup",
    "-s",
    is_flag=True,
    help="supervised flag"
)
@click.option(
    "--qtemp",
    "-qt",
    default="",
    type=str,
    help="template for query"
)
@click.option(
    "--anstemp",
    "-at",
    default="",
    type=str,
    help="tempate for response"
)
@click.option(
    "--beams",
    "-nb",
    default=5,
    type=int,
    help="number of beams"
)
@click.option(
    "--ret_seq",
    "-r",
    default=5,
    type=int,
    help="number of return sequences"
)
@click.option(
    "--num_generations",
    "-ng",
    default=100,
    type=int,
    help=""
)
@click.option(
    "--is_flax",
    "-if",
    is_flag=True,
    help="If the model is flaxed it converst it to pytorch compbatible model and save it"
)
@click.option(
    "--en",
    "-en",
    is_flag=True,
    help="The languge of head and tails"
)
@click.option(
    "--ignore_blanks",
    "-ib",
    is_flag=True,
    help="Ignore rows which have ___ in the input_text"
)
@click.option(
    "--overwrite",
    "-o",
    is_flag=True,
    help="overwrite output directory"
)
@click.option(
    "--learning_rate",
    "-lr",
    default=0,
    type=float,
    help=""
)
@click.option(
    "--wrap",
    "-w",
    is_flag=True,
    help="Whether wrap model or not"
)
@click.option(
    "--prompt_path",
    "-pp",
    default="",
    type=str,
    help="Path to save prompt encoder"
)
@click.option(
    "--plm_base_dir",
    "-base",
    default="/home/pouramini/pret",
    type=str,
    help="Base dir for pretrained models"
)
@click.option(
    "--clear",
    "-c",
    is_flag=True,
    help="Whether append results to old ones or clear results files"
)
@click.option(
    "--epochs",
    "-e",
    default=1,
    type=int,
    help=""
)
@click.option(
    "--prompt_length",
    "-pl",
    default=5,
    type=int,
    help=""
)
@click.option(
    "--train_df_path",
    "-tn",
    default= "atomic/xIntent_en_fa_train_no_dups.tsv",
    type=str,
    help=""
)
@click.option(
    "--val_df_path",
    "-vl",
    default= "atomic/xIntent_en_fa_validation_no_dups.tsv",
    type=str,
    help=""
)
@click.option(
    "--val_set",  
    "-vs",
    default="validation",
    type=str,
    help=""
)
@click.option(
    "--tresh_score",
    "-ts",
    default=0.0,
    type=float,
    help=""
)
@click.option(
    "--nli_cat",
    "-nli",
    default=-1,
    type=int,
    help=""
)
@click.option(
    "--sel_model",
    "-sel",
    default="",
    type=str,
    help=""
)
@click.option(
    "--slow",
    "-slow",
    default=0,
    type=int,
    help="Milliseconds to delay in showing results"
)
@click.option(
    "--batch_size",
    "-bs",
    default=32,
    type=int,
    help=""
)
@click.option(
    "--emb",
    "-emb",
    is_flag=True,
    help="Whether to use embedding prompt encoder"
)
@click.option(
    "--cpu",
    "-cpu",
    is_flag=True,
    help=""
)
@click.option(
    "--freez_step",
    "-fs",
    default=0,
    type=int,
    help=""
)
@click.option(
    "--unfreez_step",
    "-ufs",
    default=0,
    type=int,
    help=""
)
@click.option(
    "--inter",
    "-inter",
    is_flag=True,
    help="Interactive mode"
)
@click.option(
    "--do_eval",
    "-eval",
    is_flag=True,
    help=""
)
def main(model_id, exp_id, path, inp_samples, cycle, frozen, sup, qtemp, anstemp, beams, ret_seq, num_generations, is_flax, en, ignore_blanks, overwrite, learning_rate, wrap, prompt_path, plm_base_dir, clear, epochs, prompt_length, train_df_path, val_df_path, val_set, tresh_score, nli_cat, sel_model, slow, batch_size, emb, cpu, freez_step, unfreez_step, inter, do_eval):
    local_rank = None
    # nli categories for evaluating predictions
    nli_group = nli_map[nli_cat] if nli_cat >= 0 else "all"
    # %%
    cfg = {}
    old_vars = set()
    old_vars.update(k for k in locals() if not k.startswith('_'))
    # %%
    new_vars = set(k for k in locals() if not k.startswith('_'))
    cfg_vars = new_vars-old_vars
    cfg = {k:v for k,v in locals().items() if k in cfg_vars }
    if val_set == "train":
        # generate output samples for each input_sample in training set
        num_generations = inp_samples

    if model_id == "path":
        # load the model from the current directory
        underlying_model_name = path
    else:
        underlying_model_name = f"{plm_base_dir}/{model_id}/"

    if not Path(underlying_model_name).exists():
        confirm = "y" #input(underlying_model_name + " doesn't exists, do you want to download it?")
        if confirm != "y":
            return
        underlying_model_name = model_id

    if sup: #supervised training
        if qtemp == "": qtemp = "{event}"
        if anstemp == "": anstemp = "{response}"
    else:
        if qtemp == "": qtemp = "{rel} {event} {ph}"
        if anstemp == "": anstemp = "{ph} {response} {end}"
    if not frozen and learning_rate == 0: learning_rate = 0.0001 #6.25e-05
    if frozen and learning_rate == 0: learning_rate = 0.01  #6.25e-05
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("qtemp:", qtemp)
    print("anstemp:", anstemp)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    #iiiii create output name based on some parameters
    frozen_str = "_frozen" if frozen else "_UNfrozen"
    sup_str = "_SUP" if sup else "_UNsup"
    wrap_str = "_Wrapped" if wrap else "_UNwrapped"
    inps_num_str = "_" + str(inp_samples).replace("000","k")
    if not prompt_path and wrap and sel_model:
        prompt_path = f"sels/{sel_model}/prompt"
        print("Prompt path:", prompt_path)
    Path(prompt_path).mkdir(exist_ok=True, parents=True)
    Path("data/").mkdir(exist_ok=True, parents=True)
    split_col={"train":{}, "validation":{}}
    if not en and model_id in ["t5-base", "t5-large"]:
        en = True
    lang = "en" if en else "fa"
    if lang == "en": #langauge is english
        split_col["train"]["input_text"]="input_text" 
        split_col["train"]["targets"]=["target_text", "pred_text1", "all_preds"]
        split_col["validation"]["input_text"]="input_text"
        split_col["validation"]["targets"]=["target_text"] 
    else:
        split_col["train"]["input_text"]="input_text_fa" 
        split_col["train"]["targets"]=["target_text_fa", "pred_text1", "all_preds"]
        split_col["validation"]["input_text"]="input_text_fa"
        split_col["validation"]["targets"]=["target_text_fa"] 
    weight_decay = 0.01
    #% creating save folder
    log_dir = 'plogs/'
    train_folder=split_col["train"]["input_text"] + "--"
    for x in split_col["train"]["targets"]:
        train_folder += "=" + x
    val_folder=split_col["validation"]["input_text"] + "--"
    for x in split_col["validation"]["targets"]:
        val_folder += "=" + x
    model_name = f"{model_id}_{lang}_plength_{prompt_length}_lr_{learning_rate}{frozen_str}{sup_str}{wrap_str}{inps_num_str}_{exp_id}"
    if not sel_model:
        model_path = os.path.join(model_id,train_folder, val_folder, model_name)
        if model_id != "path":
            save_path= os.path.join(log_dir, model_path)
            ans = "y" if overwrite else ""
            while Path(save_path).exists() and ans != "y":
                ans = input(f"The {save_path} already exists, do you want to overwrite it? (y/n/other(suffix):")
                if ans == "n":
                    return
                elif ans != "y":
                    save_path += "_" + ans
        else:
            save_path = path
    else:
        save_path = f"sels/{sel_model}"
    Path(save_path).mkdir(exist_ok=True, parents=True)
    #Cut down memory usage by accumulating tiny steps for multiple backwards;
    #Should be divided exactly by batch_size
    accumulation_tiny_steps = 1 
    shuffle = False
    shuffle_evaluation=False
    validation_size = 1000
    validation_num_generation = 10
    ddp = local_rank is not None
    device = "cpu" if cpu else "cuda"
    # %% tokenizer & model
    if "mt5" in model_id:
        tokenizer = MT5TokenizerFast.from_pretrained(underlying_model_name)
        model = MT5ForConditionalGeneration.from_pretrained(underlying_model_name)
    else:
        tokenizer = T5TokenizerFast.from_pretrained(underlying_model_name)
        if is_flax:
            model = T5ForConditionalGeneration.from_pretrained(underlying_model_name, from_flax=True)
            print("converting to ", underlying_model_name[:-1] + "X")

            model.save_pretrained(underlying_model_name[:-1] + "X")
            tokenizer.save_pretrained(underlying_model_name[:-1] + "X")
            return
        else:
            model = T5ForConditionalGeneration.from_pretrained(underlying_model_name)
    #dataset_path = "../../data/v4_atomic_all_agg.csv"
    id_offset = len(tokenizer)
    encoder_relation_mappings, decoder_relation_mappings = map_relations(id_offset, length)
# %% load atomic data

    data_df = {}
    data_df["train"] = pd.read_table(train_df_path) #, index_col=[0])
    data_df["validation"] = pd.read_table(val_df_path) #, index_col=[0])
    print("Train:", len(data_df["train"]))
    print("Val:", len(data_df["validation"]))
    iterations = 0 # it's calculated based on num_samples before training 
    # eeeeee
    if do_eval:
        model.to(device=device)
        eval(model, data_df, val_set, num_generations, inter, save_path)  
        return

   #mmmmm
    def my_load_dataset(split_df, split, targets, input_text, inp_samples=0, ignore_blanks=False):
        data_split = {}
        if inp_samples == 0: inp_samples = len(split_df)
        ii = 0
        split_df = split_df.sort_values(by="input_text")
        for col in targets:
            if col in split_df:
                split_df[col] = split_df[col].astype(str)
        if ignore_blanks: # and len(split_df) > num_rows:
            split_df = split_df[split_df["input_text"].str.contains('___')==False]
        if tresh_score > 0 and "pred1_score" in split_df:
            split_df = split_df[split_df["pred1_score"] > tresh_score]
            print("*** Filtered based on pred1 score higher than ", tresh_score)
        if nli_group != "all" and "nli_group" in split_df:
            split_df = split_df[split_df["nli_group"] == nli_group]
            print("*** Filtered based on nli_group ", nli_group)

        for index, row in split_df.iterrows():
            rel = row["prefix"]
            event = row[input_text]
            if event not in data_split:
                ii += 1
                if ii > inp_samples:
                    print("input samples:", ii)
                    break
                data_split[event] = {"event":event, 'split':split}
            if not rel in data_split[event]:
                data_split[event][rel] = []
            for col in targets:
                if (col in row and len(row[col])>0):
                    target_vals = row[col].split("<br />")
                    for trg in target_vals:
                        if not trg in data_split[event][rel]:
                            data_split[event][rel].append(trg)
            #didn't convert ___ to <blank>
            #didn't normalize to lowercase
        return list(data_split.values())

    # atomic_dataset = load_atomic_dataset(dataset_path)
    #atomic_dataset = load_dataset("atomic")
    #emmm
    atomic_dataset = {}
    for split, split_data in data_df.items():
        print("split:", split)
        atomic_dataset[split] = my_load_dataset(split_data, split, split_col[split]["targets"], split_col[split]["input_text"], inp_samples=inp_samples, ignore_blanks=ignore_blanks)


    # %% dpp initialize
    is_main_process = (not ddp or local_rank == 0) 
    if ddp:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        print("launch process",local_rank)
        world_size = torch.distributed.get_world_size()
    for param in model.parameters():
        param.requires_grad = not frozen
    allowed_out_token_length = len(tokenizer)
    def clip_logits(logits):
        return logits[:,:,:allowed_out_token_length]
    clip_logits_hook = model.get_output_embeddings().register_forward_hook(
        lambda m,i,o:clip_logits(o)
    )
    # add new tokens
    #added_tokens = [ 
    #     AddedToken(token,lstrip=True,
    #         rstrip=False)
    #     for token in 
    #         list(atomic_relation_mappings.values())
    #]
    #tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
    #model.resize_token_embeddings(len(tokenizer))
    embedding_dim = model.config.hidden_size
    print("embedding dim:", embedding_dim)
    wrapped_models = {}
#ppppppp
    for rel in atomic_relation_prompt_lengths:
        prompt_encoder = None
        decoder_prompt_encoder = None
        if emb:
            prompt_encoder = EmbeddingPromptEncoder(length,embedding_dim,id_offset)
            decoder_prompt_encoder = EmbeddingPromptEncoder(1,embedding_dim,id_offset+length)
        else:
            prompt_encoder = LSTMEmbeddingPromptEncoder(length,embedding_dim,id_offset)
            decoder_prompt_encoder = LSTMEmbeddingPromptEncoder(1,embedding_dim,id_offset+length)

        added_tokens = [ 
            AddedToken(f"<{rel}_{i}>",lstrip=True,
                rstrip=False)
            for i in 
                range(length + 1)
        ]
        added_tokens += [AddedToken(token, lstrip=True, rstrip=True) for token in lang_tokens]
        tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
        model.resize_token_embeddings(len(tokenizer))
        if wrap:
            wrapped_models[rel] = PTuningWrapper(model,prompt_encoder,decoder_prompt_encoder,prompt_token_fn=get_prompt_token_fn(id_offset,length))
        else:
            wrapped_models[rel] = model
        #model.resize_token_embeddings(len(tokenizer))
        print(f"Tokenizer size: {len(tokenizer)}")
        #wrapped_model,prompt_func,prompt_string = PTuningWrapper.\
        #    interval_prompt(
        #        model,tokenizer,(2,3,0),(0,0,0),return_prompt_string=True
        #    )
        #print("Prompt string:",prompt_string)
        #example = prompt_func("piece one","piece two")
        #print("Example:",example)
        if prompt_path:
            wrapped_models[rel].prompt_encoder.load(prompt_path)
        #if wrap:
        #    wrapped_models[rel] = wrapped_model
        #else:
        #    wrapped_models[rel] = model
        #model.resize_token_embeddings(len(tokenizer))
        # %% Aggregate instances of queries and corresponding responses
    # (str)split_name -> (dict) query -> (list) response 
    print("building query responses")
    atomic_query_responses = {}
    # ttttt
    ss = 0
    for split_name,split_df in atomic_dataset.items():
        atomic_query_responses[split_name] = {}
        for row in split_df:
            for rel in atomic_relation_mappings:
                if rel not in atomic_query_responses[split_name]:
                    atomic_query_responses[split_name][rel] = {}
                if len(row[rel])>0: 
                    encoder_rel_tokens = encoder_relation_mappings[rel]
                    decoder_rel_tokens = decoder_relation_mappings[rel]
                    gen_token = gen_token_fa if lang == "fa" else gen_token_en
                    #query = f"{rel_tokens} {row['event']}" #Change this line to modify the encoder input
                    query = qtemp.format(gen=gen_token, event=row['event'], rel=encoder_rel_tokens, ph=placeholder_token) #Change this line to modify the encoder input
                    # query = f"{row['event']} {rel_tokens}" #Change this line to modify the encoder input
                    if query not in atomic_query_responses[split_name][rel]:
                        atomic_query_responses[split_name][rel][query] = []
                    for response in row[rel]:
                        answer = anstemp.format(gen=gen_token, response=response, rel=decoder_rel_tokens, ph=placeholder_token, end="<extra_id_1>")
                        if ss < 3:
                            print(query)
                            print(answer)
                            ss+=1 
                        atomic_query_responses[split_name][rel][query].append((answer,response))
                            #Change this line to modify the decoder input
                    #didn't convert ___ to <blank>
                    #didn't normalize to lowercase

    #flatten
    print("building flattened pairs")
    atomic_flattened = {}
    kk = 0
    for split_name,rel_queries_responses in atomic_query_responses.items():
        atomic_flattened[split_name] = {}
        for rel,queries_responses in rel_queries_responses.items():
            atomic_flattened[split_name][rel] = []
            for query,responses in queries_responses.items():
                for response,_ in responses:
                    atomic_flattened[split_name][rel].append((query,response))
                    kk +=1

    iterations = kk 
    print("len flattened:", len(atomic_flattened["train"]), "iterations:", kk)
#llllll
# %% Prepare training data
    def collate_fn_for_flattened(batch):
        queries,responses = zip(*batch)
        new_batch = tokenizer(list(queries),return_tensors='pt',padding='longest')
        with tokenizer.as_target_tokenizer():
            outputs = tokenizer(list(responses),return_tensors='pt',padding='longest')
            labels = outputs['input_ids']
            labels[labels==tokenizer.pad_token_id] = -100
            new_batch['labels']=labels
            new_batch['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(
                outputs['input_ids']
            )
            new_batch['decoder_attention_mask'] = outputs['attention_mask']
        return new_batch

    # def collate_fn_for_generation(batch):
    #     queries,references = zip(*batch)
    #     new_batch = tokenizer(queries,return_tensors='pt',padding='longest')
    #     return new_batch,references
# %% build dataloader
    # %% dataloader and  parallel
    node_batch_size = batch_size//accumulation_tiny_steps
    train_sampler = {}
    train_dataloader = {}
    dev_dataloader = {}
    for rel in atomic_relation_mappings:
        train_sampler[rel] = None
        if shuffle:
            train_sampler[rel] = torch.utils.data.RandomSampler(atomic_flattened['train'][rel])
        if ddp:
            assert node_batch_size%world_size == 0
            node_batch_size = node_batch_size//world_size
            train_sampler[rel] = torch.utils.data.DistributedSampler(atomic_flattened['train'][rel],shuffle=shuffle)
        train_dataloader[rel] = torch.utils.data.DataLoader(atomic_flattened['train'][rel],
            batch_size=node_batch_size,sampler=train_sampler[rel],
            collate_fn=collate_fn_for_flattened)
        if is_main_process:
            print("dev data loader")
            dev_dataloader[rel] = torch.utils.data.DataLoader(atomic_flattened['validation'][rel],
                batch_size=node_batch_size,shuffle=shuffle_evaluation,
                collate_fn=collate_fn_for_flattened)

    # %% prepare for training
    if is_main_process:
        print("init sw to: ", save_path) 
        sw = SummaryWriter(save_path)
        serialization_dir = save_path
        tokenizer.save_pretrained(serialization_dir)
        with open(os.path.join(serialization_dir,'exp_config.json'),'w') as f:
            json.dump(cfg,f,ensure_ascii=False,indent=4)
    for wrapped_model in wrapped_models.values():
        wrapped_model.to(device=device)

    if ddp:
        for rel in wrapped_models:
            wrapped_models[rel] = torch.nn.parallel.DistributedDataParallel(wrapped_models[rel],device_ids=[local_rank])
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]


    # %%
    warm_up_steps = 0.002*iterations
    for rel,wrapped_model in wrapped_models.items():
        wrapped_model.zero_grad()
        optimizer_grouped_parameters = [
            {"params":[p for p in wrapped_model.parameters() if p.requires_grad]}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate,eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,warm_up_steps,iterations)
        step = 0
        done = False
        best_dev_loss = 1e10
        train_iter = iter(train_dataloader[rel])
        if is_main_process:
            pbar = tqdm(total=iterations*epochs,dynamic_ncols=True,desc=rel)
        # wwww
        for epoch in range(epochs):
            step = 0
            while step <= iterations:
                try:
                    if False and is_main_process and (step % cycle == 0 and step > 0): #validation
                        #print("start validating...")
                        with torch.no_grad():
                            if wrap:
                                if ddp:
                                    wrapped_model.module.update_model_weight()
                                else:
                                    wrapped_model.update_model_weight()
                            if frozen:
                                for p in model.parameters():
                                    p.requires_grad = False 
                            model.eval()
                            pbar.set_description(f'validating...{rel}')
                            dev_allset_micro_loss = 0.
                            dev_token_loss = 0.
                            dev_token_count = 0
                            dev_sample_loss = 0. #avg on sample
                            dev_sample_count = 0
                            for batch in tqdm(dev_dataloader[rel],desc=f'validating ...',leave=False):
                                if dev_sample_count>=validation_size:
                                    break
                                batch = {k:v.to(device=device) for k,v in batch.items()}
                                result = model(**batch)
                                # logits = clip_logits(result['logits'])
                                logits = result['logits']
                                loss = torch.nn.functional.cross_entropy(
                                    logits.reshape(-1,logits.size(2)),
                                    batch['labels'].reshape(-1,),
                                    reduction='none'
                                ).reshape(logits.size(0),-1)
                                labels_mask = (batch['labels'] != -100) 
                                dev_token_loss += loss.sum().item()
                                dev_token_count += labels_mask.sum().item()
                                dev_sample_loss += (loss.sum(dim=-1)/labels_mask.sum(dim=-1)).sum().item()
                                dev_sample_count += logits.size(0)
                                del result
                                del loss
                                del labels_mask
                            dev_micro_avg_loss = dev_token_loss/dev_token_count
                            dev_macro_avg_loss = dev_sample_loss/dev_sample_count
                            sw.add_scalar(f'dev/{rel}/micro_avg_loss',dev_micro_avg_loss,step)
                            sw.add_scalar(f'dev/{rel}/macro_avg_loss',dev_macro_avg_loss,step)
                            if dev_micro_avg_loss < best_dev_loss:
                                best_dev_loss = dev_micro_avg_loss
                                model.save_pretrained(serialization_dir)
                            #ggg
                            if step == iterations:
                                generation_results = \
                                "|Queries|Generation Results| Target |\n"\
                                "|-|-|-|\n"
                                for i,(query,responses) in enumerate(tqdm(atomic_query_responses['validation'][rel].items())):
                                #for i,key in enumerate(tqdm(atomic_query_responses['validation'][rel])):
                                    if i==validation_num_generation:
                                        break
                                    results = tokenizer.batch_decode(
                                        model.generate(**tokenizer(query, return_tensors='pt').to(device=device),**generation_params),
                                        skip_special_tokens=True
                                    )
                                    generation_results+=f"|`{query}`|`{str(results)}`|`{str(responses)}`|\n"
                                sw.add_text(f'dev/{rel}/generation_samples',generation_results,step)
                    # end validation
                    if unfreez_step > 0 and step > unfreez_step and frozen and not done:
                        print("unfreezing the model")
                        done = True
                        for p in model.parameters():
                            p.requires_grad = True # Unfreezing
                    if freez_step > 0 and step > freez_step and not frozen and not done:
                        print("freezing the model")
                        done = True
                        for p in model.parameters():
                            p.requires_grad = False # freezing
                    model.train()
                    optimizer.zero_grad()
                    batch_loss = torch.tensor(0.)
                    for tiny_step in range(accumulation_tiny_steps):
                        try:
                            batch = next(train_iter)
                        except StopIteration:
                            train_iter = iter(train_dataloader[rel])
                            batch = next(train_iter)
                        batch = {k:v.to(device=device) for k,v in batch.items()}
                        result = wrapped_model(**batch)
                        loss = result['loss']/accumulation_tiny_steps
                        #logits = clip_logits(result['logits'])
                        #loss = torch.nn.functional.cross_entropy(
                        #     logits.reshape(-1,logits.size(2)),
                        #     labels.reshape(-1,)
                        #)/accumulation_tiny_steps
                        loss.backward()
                        batch_loss += loss.item()
                    optimizer.step()
                    scheduler.step()
                    step+=1
                    if ddp:
                        # loss = loss.detach()
                        losses = [torch.zeros_like(batch_loss) for i in range(world_size)]
                        torch.distributed.all_gather(tensor_list=losses,tensor=batch_loss)
                        batch_loss = torch.stack(losses).mean()
                    if is_main_process:
                        pbar.set_description(f'training...{rel}')
                        pbar.update()
                        sw.add_scalar(f'train/{rel}/loss',batch_loss.item(),global_step=step)
                    del result
                    del loss
                except KeyboardInterrupt:
                    print("exiting while ...")
                    break
            # end train while
        # end epochs loop
        # eww
        if is_main_process:
            pbar.close()
# %%

    model.save_pretrained(serialization_dir)
    for rel, wrapped_model in wrapped_models.items():
        if prompt_path and sel_model and wrap:
            print(">>> saving prompt encoder")
            wrapped_model.prompt_encoder.save(prompt_path)
        if wrap:
            with torch.no_grad():
                if ddp:
                    wrapped_model.module.update_model_weight()
                else:
                    wrapped_model.update_model_weight()
        eval(model, data_df, val_set, num_generations, inter, save_path)  
    #device = 'cuda:0'
    #model = model.to(device)
# %%
if __name__ == "__main__":
    main()
