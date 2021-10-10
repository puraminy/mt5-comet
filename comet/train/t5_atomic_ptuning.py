#%% load libraries

from datasets import load_dataset,load_metric
from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast, AdamW, AddedToken,
    MT5ForConditionalGeneration, MT5TokenizerFast,
    get_linear_schedule_with_warmup
)
from transformers_ptuning import PTuningWrapper
import torch
import csv 
import re
import json
from torch.utils.tensorboard import SummaryWriter
import os,time
import argparse
from tqdm.auto import tqdm
from transformers_ptuning.ptuning_wrapper import LSTMEmbeddingPromptEncoder, EmbeddingPromptEncoder
#%% argparse
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
    default=5000,
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
    help=""
)
@click.option(
    "--sup",
    "-s",
    is_flag=True,
    help="supervised flag"
)
def main(model_id, exp_id, path, iterations, cycle, frozen, sup):
    local_rank = None
    #%%
    cfg = {}
    old_vars = set()
    old_vars.update(k for k in globals() if not k.startswith('_'))
    #%% some hyper-parameters
    if not model_id == "path":
        underlying_model_name = f"/drive2/pretrained/mt5/hf/{model_id}/"
    else:
        underlying_model_name = path
    #underlying_model_name = "logs/mt5-small/prompt_length_3/last"
    if sup:
        learning_rate = 0.0001 #6.25e-05
    else:
        learning_rate = 0.01 #6.25e-05
    #iiiii
    frozen_str = "_frozen" if frozen else "_UNfrozen"
    sup_str = "_SUP" if sup else "_UNsup"
    prompt_length = 5
    split_col={"train":{}, "validation":{}}
    split_col["train"]["input_text"]="input_text_fa"
    split_col["train"]["target_text"]="target_text_fa"
    split_col["validation"]["input_text"]="input_text_fa"
    split_col["validation"]["target_text"]="target_text_fa"
    warm_up_steps = 0.002*iterations
    weight_decay = 0.01
    batch_size = 16
    #% creating save folder
    log_dir = '/drive2/pouramini/plogs/'
    train_folder=split_col["train"]["input_text"] + "--" + split_col["train"]["target_text"]
    val_folder=split_col["validation"]["input_text"] + "--" + split_col["validation"]["target_text"]
    model_name = os.path.join(model_id,train_folder, val_folder,f"plength_{prompt_length}_lr_{learning_rate}{frozen_str}{sup_str}_{exp_id}")
    if model_id != "path":
        save_path= os.path.join(log_dir,model_name)
        ans = ""
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
        "num_beams":5,
        "num_return_sequences":5,
    }
    ddp = local_rank is not None
    device = 'cuda'
    dataset_path = "/home/pouramini/mt5-comet/data/v4_atomic_all_agg.csv"
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
    #%%
    atomic_relation_prompt_lengths = {
        "xIntent":prompt_length,
    }
    #%%
    new_vars = set(k for k in globals() if not k.startswith('_'))
    cfg_vars = new_vars-old_vars
    cfg = {k:v for k,v in globals().items() if k in cfg_vars }
    #%% load atomic data

    import pandas as pd
    data_paths = {}
    #data_paths["train"] = pd.read_table("/home/pouramini/atomic/xIntent_en_train_no_dups.tsv")
    #data_paths["validation"] = pd.read_table("/home/pouramini/atomic/xIntent_en_fa_validation_no_dups.tsv")

    data_paths["train"] = pd.read_table("/drive3/pouramini/data/atomic/en_fa/xIntent_en_fa_train_no_dups.tsv")
    data_paths["validation"] = pd.read_table("/drive3/pouramini/data/atomic/en_fa/xIntent_en_fa_validation_no_dups.tsv")

    def my_load_dataset(split_data, split, target_text, input_text):
        data_split = {}
        split_data[target_text] = split_data[target_text].astype(str)
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
    for split, split_data in data_paths.items():
        print("split:", split)
        atomic_dataset[split] = my_load_dataset(split_data, split, split_col[split]["target_text"], split_col[split]["input_text"])

    placeholder_token = "<extra_id_0>"
    #%% dpp initialize
    is_main_process = (not ddp or local_rank == 0) 
    if ddp:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        print("launch process",local_rank)
        world_size = torch.distributed.get_world_size()
    #%% tokenizer & model
    if "mt5" in model_id:
        tokenizer = MT5TokenizerFast.from_pretrained(underlying_model_name)
        model = MT5ForConditionalGeneration.from_pretrained(underlying_model_name)
    else:
        tokenizer = T5TokenizerFast.from_pretrained(underlying_model_name)
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
    # added_tokens = [ 
    #     AddedToken(token,lstrip=True,
    #         rstrip=False)
    #     for token in 
    #         list(atomic_relation_mappings.values())
    # ]
    # tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
    # model.resize_token_embeddings(len(tokenizer))
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
        added_tokens = [ 
            AddedToken(f"<{rel}_{i}>",lstrip=True,
                rstrip=False)
            for i in 
                range(length)
        ]
        tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
        wrapped_models[rel] = PTuningWrapper(model,prompt_encoder,prompt_token_fn=get_prompt_token_fn(id_offset,length))
    #%% Aggregate instances of queries and corresponding responses
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
                    if sup:
                        query = f"{d['event']} " #Change this line to modify the encoder input
                    else:
                        query = f"{rel_tokens} {d['event']} because PersonX intended {placeholder_token}" #Change this line to modify the encoder input
                    # query = f"{d['event']} {rel_tokens}" #Change this line to modify the encoder input
                    if query not in atomic_query_responses[split_name][rel]:
                        atomic_query_responses[split_name][rel][query] = []
                    for response in d[rel]:
                        if sup:
                            atomic_query_responses[split_name][rel][query].append((f"{response}",response)) #Change this line to modify the decoder input
                        else:
                            atomic_query_responses[split_name][rel][query].append((f"{placeholder_token} {response} <extra_id_1>",response)) #Change this line to modify the decoder input
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
    #%% dataloader and  parallel
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


    #%%
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
    num_generations = 0
    for rel in ["xIntent"]:
        for i,(query,responses) in enumerate(tqdm(atomic_query_responses[val_set][rel].items())):
            if num_generations>0 and i>= num_generations:
                break
            hyps = tokenizer.batch_decode(
                            model.generate(**tokenizer(query, return_tensors='pt').to(device=device),**generation_params),
                            skip_special_tokens=True
                        )
            query = re.sub(r'<.*?>','',query)
            tails = [x[1] for x in responses]
            results.append({
                "head":query,
                "gens":hyps,
                "tails":tails,
            })
        #%%
    with open(os.path.join(save_path,f"{val_set}_gen.json"),'w') as f:
        json.dump(results,f,ensure_ascii=False,indent=2)
    # %%
if __name__ == "__main__":
    main()
