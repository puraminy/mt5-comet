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

from sentence_transformers import SentenceTransformer, util
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
    "--iterations",
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
    default="",
    type=str,
    help="template for query"
)
@click.option(
    "--anstemp",
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
    "-g",
    default=200,
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
    default="",
    type=str,
    help="Path to save prompt encoder"
)
@click.option(
    "--plm_base_dir",
    "-base",
    default="",
    type=str,
    help="Base dir for pretrained models"
)
def main(model_id, exp_id, path, iterations, cycle, frozen, sup, qtemp, anstemp, beams, ret_seq, num_generations, is_flax, en, ignore_blanks, overwrite, learning_rate, wrap, prompt_path):
    local_rank = None
    if prompt_path:
        Path(prompt_path).mkdir(exist_ok=True, parents=True)
    # %%
    cfg = {}
    old_vars = set()
    old_vars.update(k for k in locals() if not k.startswith('_'))
    # %% some hyper-parameters
    if not model_id == "path":
        underlying_model_name = f"{plm_base_dir}/{model_id}/"
    else:
        underlying_model_name = path
    if not Path(underlying_model_name).exists():
        confirm = input(underlying_model_name + " doesn't exists, do you want to download it?")
        if confirm != "y":
            return
        underlying_model_name = model_id

    #underlying_model_name = "logs/mt5-small/prompt_length_3/last"
    if sup:
        if qtemp == "": qtemp = "{event}"
        if anstemp == "": anstemp = "{response}"
        if learning_rate == 0: learning_rate = 0.0001 #6.25e-05
    else:
        if qtemp == "": qtemp = "{rel} {event} {ph}"
        if anstemp == "": anstemp = "{ph} {response} {end}"
        if learning_rate == 0: learning_rate = 0.01  #6.25e-05
    #iiiii
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("qtemp:", qtemp)
    print("anstemp:", anstemp)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    frozen_str = "_frozen" if frozen else "_UNfrozen"
    sup_str = "_SUP" if sup else "_UNsup"
    wrap_str = "_Wrapped" if wrap else "_UNwrapped"
    iter_str = "_" + str(iterations).replace("000","k")
    prompt_length = 5
    split_col={"train":{}, "validation":{}}
    split_col["train"]["input_text"]="input_text" if en else "input_text_fa"
    split_col["train"]["target_text"]="target_text" if en else "target_text_fa"
    split_col["validation"]["input_text"]="input_text" if en else "input_text_fa"
    split_col["validation"]["target_text"]="target_text" if en else "target_text_fa"
    warm_up_steps = 0.002*iterations
    weight_decay = 0.01
    batch_size = 16
    #% creating save folder
    log_dir = 'plogs/'
    train_folder=split_col["train"]["input_text"] + "--" + split_col["train"]["target_text"]
    val_folder=split_col["validation"]["input_text"] + "--" + split_col["validation"]["target_text"]
    model_name = f"plength_{prompt_length}_lr_{learning_rate}{frozen_str}{sup_str}{wrap_str}{iter_str}_{exp_id}"
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
    #Cut down memory usage by accumulating tiny steps for multiple backwards;
    #Should be divided exactly by batch_size
    accumulation_tiny_steps = 1 
    shuffle = True
    shuffle_evaluation=False
    validation_size = 1000
    validation_num_generation = 10
    generation_params = {
        "max_length":80,
        "early_stopping":True,
        "num_beams":beams,
        "num_return_sequences":ret_seq,
    }
    ddp = local_rank is not None
    device = 'cuda'
    #dataset_path = "../../data/v4_atomic_all_agg.csv"
    atomic_relation_prompt_lengths = {
        "xIntent":prompt_length,
        "oEffect":prompt_length,
        "oReact":prompt_length,
        "oWant":prompt_length,
        "xAttr":prompt_length,
        "xEffect":prompt_length,
        "xNeed":prompt_length,
        "xReact":prompt_length,
        "xWant":prompt_length
    }
    # %%
    atomic_relation_prompt_lengths = {
        "xIntent":prompt_length,
    }
    # %%
    new_vars = set(k for k in locals() if not k.startswith('_'))
    cfg_vars = new_vars-old_vars
    cfg = {k:v for k,v in locals().items() if k in cfg_vars }
# %% load atomic data

    data_df = {}
    #data_df["train"] = pd.read_table("/home/pouramini/atomic/xIntent_en_train_no_dups.tsv")
    #data_df["validation"] = pd.read_table("/home/pouramini/atomic/xIntent_en_fa_validation_no_dups.tsv")
    train_df_path = "/home/pouramini/atomic/xIntent_en_fa_train_no_dups.tsv"

    if Path("data/train_" + str(iterations) + ".tsv").exists():
        print("***************** Loading ... saved data")
        train_df_path= "data/train_" + str(iterations) + ".tsv" 
        ignore_blanks = False
    val_df_path = "/home/pouramini/atomic/xIntent_en_fa_validation_no_dups.tsv"
    if Path("data/validation_" + str(num_generations) + ".tsv").exists():
        print("***************** Loading ... saved validation data")
        val_df_path= "data/validation_" + str(num_generations) + ".tsv"
        ignore_blanks = False

    data_df["train"] = pd.read_table(train_df_path)
    data_df["validation"] = pd.read_table(val_df_path)

    def my_load_dataset(split_data, split, target_text, input_text, num_rows=0, ignore_blanks=False):
        data_split = {}
        split_data[target_text] = split_data[target_text].astype(str)
        if ignore_blanks:
            split_data = split_data[split_data["input_text"].str.contains('___')==False]
        if num_rows > 0:
            print(len(split_data), " ? ", num_rows)
            assert len(split_data) >= num_rows, "Not enough rows for " + str(num_rows)
            split_data = split_data.head(num_rows)
            Path("data").mkdir(exist_ok=True)
            print(">>>>> Saving data")
            split_data.to_csv("data/" + split + "_" + str(num_rows)+ ".tsv", sep="\t")
        
        for index, d in split_data.iterrows():
            rel = d["prefix"]
            if len(d[target_text])>0: 
                event = d[input_text]
                if event not in data_split:
                    data_split[event] = {"event":event, 'split':split}
                if not rel in data_split[event]:
                    data_split[event][rel] = []
                data_split[event][rel].append(d[target_text])
                #didn't convert ___ to <blank>
                #didn't normalize to lowercase
        return list(data_split.values())

    def load_atomic_dataset(path):
        data={}
        with open(path) as source_file:
            source_reader = csv.reader(source_file)
            # Read first line to get column name
            source_line = next(source_reader)
            event_colname = source_line[0]
            categories_colname = source_line[1:10]
            prefix_colname = source_line[10]
            split_colname = source_line[11]
            for source_line in source_reader:
                # get every column
                event = source_line[0]
                annotationss = [
                    json.loads(raw_anns) for raw_anns in source_line[1:10]]
                event_prefix = source_line[10]
                event_split = source_line[11]
                if event_split not in data:
                    data[event_split] = []
                d = {"event":event}
                d.update({category:annotations for 
                    category,annotations in zip(categories_colname,annotationss)})
                data[event_split].append(d)
        return data

    # atomic_dataset = load_atomic_dataset(dataset_path)
    #atomic_dataset = load_dataset("atomic")
    atomic_dataset = {}
    for split, split_data in data_df.items():
        print("split:", split)
        num_rows = iterations if split == "train" else num_generations
        atomic_dataset[split] = my_load_dataset(split_data, split, split_col[split]["target_text"], split_col[split]["input_text"], num_rows=num_rows, ignore_blanks=ignore_blanks)

    placeholder_token = "<extra_id_0>"
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
    gen_token = "<gen>"
    # %% dpp initialize
    is_main_process = (not ddp or local_rank == 0) 
    if ddp:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        print("launch process",local_rank)
        world_size = torch.distributed.get_world_size()
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

    for param in model.parameters():
        param.requires_grad = not frozen
    allowed_out_token_length = len(tokenizer)
    def clip_logits(logits):
        return logits[:,:,:allowed_out_token_length]
    #可以在这里就加钩子，因为模型不会替换模块，只会更新权重
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
    atomic_relation_mappings = {}
    def get_prompt_token_fn(id_offset,length):
        return lambda x: (x>=id_offset)&(x<id_offset+length)
    for rel in atomic_relation_prompt_lengths:
        id_offset = len(tokenizer)
        print("id_offset:", id_offset)
        length = atomic_relation_prompt_lengths[rel]
        atomic_relation_mappings[rel] = " ".join(f"<{rel}_{i}>" for i in range(length))
        prompt_encoder = LSTMEmbeddingPromptEncoder(length,embedding_dim,id_offset)
        if prompt_path:
            prompt_encoder.load(prompt_path)
        added_tokens = [ 
            AddedToken(f"<{rel}_{i}>",lstrip=True,
                rstrip=False)
            for i in 
                range(length)
        ]
        tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
        model.resize_token_embeddings(len(tokenizer))
        if wrap:
            wrapped_models[rel] = PTuningWrapper(model,prompt_encoder,prompt_token_fn=get_prompt_token_fn(id_offset,length))
        else:
            wrapped_models[rel] = model
        # %% Aggregate instances of queries and corresponding responses
    # (str)split_name -> (dict) query -> (list) response 
    print("building query responses")
    atomic_query_responses = {}
    # ttt
    for split_name,split_data in atomic_dataset.items():
        atomic_query_responses[split_name] = {}
        for d in split_data:
            for rel in atomic_relation_mappings:
                if rel not in atomic_query_responses[split_name]:
                    atomic_query_responses[split_name][rel] = {}
                if len(d[rel])>0: 
                    rel_tokens = atomic_relation_mappings[rel]
                    #query = f"{rel_tokens} {d['event']}" #Change this line to modify the encoder input
                    query = qtemp.format(event=d['event'], rel=rel_tokens, ph=placeholder_token) #Change this line to modify the encoder input
                    # query = f"{d['event']} {rel_tokens}" #Change this line to modify the encoder input
                    if query not in atomic_query_responses[split_name][rel]:
                        atomic_query_responses[split_name][rel][query] = []
                    for response in d[rel]:
                        answer = anstemp.format(response=response, rel=rel_tokens, ph=placeholder_token, end="<extra_id_1>")
                        atomic_query_responses[split_name][rel][query].append((answer,response))
                            #Change this line to modify the decoder input
                    #didn't convert ___ to <blank>
                    #didn't normalize to lowercase

    #flatten
    print("building flattened pairs")
    atomic_flattened = {}
    for split_name,rel_queries_responses in atomic_query_responses.items():
        atomic_flattened[split_name] = {}
        for rel,queries_responses in rel_queries_responses.items():
            atomic_flattened[split_name][rel] = []
            for query,responses in queries_responses.items():
                for response,_ in responses:
                    atomic_flattened[split_name][rel].append((query,response))

# %% Prepare training data

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
            import json
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
    for rel,wrapped_model in wrapped_models.items():
        optimizer_grouped_parameters = [
            {"params":[p for p in wrapped_model.parameters() if p.requires_grad]}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate,eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,warm_up_steps,iterations)
        step = 0
        done = True
        best_dev_loss = 1e10
        train_iter = iter(train_dataloader[rel])
        if is_main_process:
            pbar = tqdm(total=iterations,dynamic_ncols=True,desc=rel)
        while step <= iterations:
            try:
                if is_main_process and (step % cycle == 0 and step > 0): #validation
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
                if step > 4000 and frozen and not done:
                    print("unfreezing the model")
                    done = True
                    for p in model.parameters():
                        p.requires_grad = True # Unfreezing
                if step > 4000 and not frozen and not done:
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
                    # logits = clip_logits(result['logits'])
                    # loss = torch.nn.functional.cross_entropy(
                    #     logits.reshape(-1,logits.size(2)),
                    #     batch['labels'].reshape(-1,)
                    # )/accumulation_tiny_steps
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
        if is_main_process:
            pbar.close()
# %%

    results = []
    #device = 'cuda:0'
    #model = model.to(device)
    val_set = "validation"
    scorer_model = SentenceTransformer('/home/pouramini/pret/mm/paraphrase-multilingual-MiniLM-L12-v2/')
    #sss
    df = data_df[val_set]
    target = "target_text"
    df[target] = df[target].astype(str)
    df = df.groupby(['prefix','input_text'],as_index=False)[target].agg('<br />'.join)
    print("Scoring...")
    df["pred1_score"] = 0
    df["pred_text1"] = ""
    df["top"] = ""

    for rel,wrapped_model in wrapped_models.items():
        print(">>> saving prompt encoder")
        if prompt_path:
            wrapped_model.prompt_encoder.save(prompt_path)
        if wrap:
            with torch.no_grad():
                if ddp:
                    wrapped_model.module.update_model_weight()
                else:
                    wrapped_model.update_model_weight()
        model.eval()
        sum_score = 0 
        total = num_generations
        #if num_generations == 0:
        total = min(num_generations, len(atomic_query_responses[val_set][rel].items()))
        pbar = tqdm(atomic_query_responses[val_set][rel].items(), total = total)
        for idx,(query,responses) in enumerate(pbar):
            if num_generations>0 and idx>= num_generations:
                print("break at ", idx)
                break

            hyps = tokenizer.batch_decode(
                            model.generate(**tokenizer(query, return_tensors='pt').to(device=device),**generation_params),
                            skip_special_tokens=True
                        )
            query = re.sub(r'<.*?>','',query)
            tails = [x[1] for x in responses]
  
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
            cond = (df['prefix'] == rel) & (df['input_text'] == query)
            df.loc[cond, "top"] = closest
            df.loc[cond, "pred_text1"] = pred_text
            df.loc[cond, "pred1_score"] = float("{:.4f}".format(top["score"]))
            cur_score = top["score"]
            sum_score += cur_score
            mean_score = "{:.4f}".format(sum_score / idx)
            #tqdm.write(f"Mean score:{mean_score}")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(idx, "::",query)
            print("Prediction:", pred_text)
            print("Closest tail:", closest)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            pbar.set_description(f"Mean score:{mean_score} cur score {cur_score:.2f}")
            pbar.update(1)

            results.append({
                "head":query,
                "gens":hyps,
                "tails":tails,
            })
        # %%%%%%%%%%%%%%%%%%
    df = df[df["pred1_score"] > 0]
    print("Results:", df["pred1_score"].mean())
    pbar.close()
    with open(os.path.join(save_path,f"{val_set}_gen.json"),'w') as f:
        json.dump(results,f,ensure_ascii=False,indent=2)
    # %% save dataframe %
    out = save_path + "/scored_" + model_name  + ".tsv" 
    print(out)
    print(len(df))
    df.to_csv(out, sep="\t", index=False)
    with open("/home/pouramini/dflist", "w") as dflist:
        print(f"{model_name}={out}", file=dflist)
# %%
if __name__ == "__main__":
    main()
