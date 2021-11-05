#%% load libraries
from comet.train.common import *
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



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    "--from_dir",
    "-",
    is_flag=True,
    help=""
)
@click.option(
    "--train_samples",
    "-n",
    default=100,
    type=int,
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
    "--val_samples",
    "-ng",
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
    default="/home/ahmad/pret",
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
    default="",
    type=str,
    help=""
)
@click.option(
    "--output_name",
    "-out",
    default="",
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
    default="{ph} {response} {end}",
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
    "--natural",
    "-nat",
    is_flag=True,
    help="Whether use natural (with template) input output or not"
)
@click.option(
    "--nli_group",
    "-nli",
    default="all",
    type=str,
    help="filter by predicted nli group"
)
@click.option(
    "--learning_rate",
    "-lr",
    default=0,
    type=float,
    help="learning rate"
)
@click.option(
    "--do_eval",
    "-eval",
    is_flag=True,
    help=""
)
@click.option(
    "--inter",
    "-inter",
    is_flag=True,
    help="Interactive output generation"
)
@click.option(
    "--cont",
    "-cont",
    is_flag=True,
    help="continue training"
)
@click.option(
    "--wrap",
    "-w",
    default="",
    type=str,
    help=""
)
@click.option(
    "--frozen",
    "-f",
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
    "--cpu",
    "-cpu",
    is_flag=True,
    help=""
)
@click.option(
    "--load_prompt_path",
    "-pp",
    default="",
    type=str,
    help=""
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help=""
)
@click.option(
    "--cycle",
    "-c",
    default=500,
    type=int,
    help=""
)
@click.option(
    "--batch_size",
    "-bs",
    default=8,
    type=int,
    help=""
)
def main(model_id, qtemp, anstemp, train_samples, val_set, 
         val_samples, load_path, overwrite, save_path, output_name, lang, pred_tresh, ignore_blanks, natural, nli_group, learning_rate, do_eval, inter, cont, wrap, frozen, freez_step, unfreez_step, cpu, load_prompt_path, verbose, cycle, batch_size, path, from_dir, is_flax):

    #%% some hyper-parameters
    #bbbbbbbbbbb
    #underlying_model_name = "logs/atomic-mt5/last"
    if save_path == "":
        save_path = "/content/drive/MyDrive/backup/logs"
        if "ahmad" in home:
            save_path = "/home/ahmad/logs"


    if from_dir:
        underlying_model_name = path
    elif Path(load_path).exists():
        underlying_model_name = f"{load_path}/{model_id}"
        if not Path(underlying_model_name).exists():
            underlying_model_name = model_id        
    else:
        underlying_model_name = model_id
        
    weight_decay = 0.01
    shuffle = False
    shuffle_evaluation=False
    validation_size = 100
    validation_num_generation = 20
    if not frozen and learning_rate == 0: learning_rate = 6.25e-05
    if frozen and learning_rate == 0: learning_rate = 0.01  #6.25e-05
    generation_params = {
        "max_length":80,
        "early_stopping":True
    }
    device = 'cuda' if not cpu else 'cpu'
    args = locals()

    set_device(device)
    log_dir = save_path
    output_name = model_id if not output_name else output_name
    save_path = os.path.join(log_dir, output_name)
    model_name = f"{learning_rate}_{cycle}_{train_samples}"
    mlog.info(f"SAVE Path:{save_path}")
    conf_path = os.path.join(save_path,'conf.json')
    checkpoint = None
    if Path(save_path).exists() and not model_id=="test" and (cont or do_eval):
        mlog.info("Loading from %s", save_path)
        underlying_model_name = save_path
        checkpoint = torch.load(os.path.join(save_path,"saved_states"))
        if Path(conf_path).exists():
           with open(conf_path, 'r') as f:
               args = json.load(f) 
           mlog.info(args)
           mlog.info("Loading from configuration file")
           qtemp = args['qtemp']
           atemp = args['anstemp']
           mlog.info("Qtemp: %s", args['qtemp'])
           mlog.info("Anstemp: %s", args['anstemp'])

    for logger in [mlog, vlog, clog, dlog]:
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        logger.info(f"%%%%%%%%%%%%%%%%%% { model_id } ")
        logger.info(f"%%%%%%%%%%%%%%%%%% { output_name } ")
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    args_str = json.dumps(args, indent=4)
    for logger in [clog, vlog]:
        logger.info(args_str)
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    ii = 1
    while not overwrite and Path(save_path).exists() and not model_id=="test":
        ans = input("The output directory already exists, do you want to overwite it? (y/n)")
        if ans == "y":
            overwrite = True
        save_path = os.path.join(log_dir,output_name + "_"+str(ii))
        mlog.info(save_path)
        ii += 1

    Path(save_path).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(save_path, "best_model")).mkdir(exist_ok=True, parents=True)
    with open(conf_path, 'w') as outfile:
        json.dump(args, outfile, indent=4)
    #%% load atomic data
    import pandas as pd
    atomic_dataset = {}
    train_path= "atomic/xIntent_en_fa_train_no_dups.tsv"
    val_path= "atomic/xIntent_en_fa_validation_no_dups.tsv"
    atomic_dataset["train"] = pd.read_table(train_path)
    atomic_dataset["validation"] = pd.read_table(val_path)

    atomic_query_responses = {}
    atomic_flattened = {}
    num_records = {}
    num_samples = {"train": train_samples, "validation":val_samples}
    for split_name,split_df in atomic_dataset.items():
        (atomic_query_responses[split_name], 
         atomic_flattened[split_name],
         num_records[split_name]
        )= fill_data(split_df, split_name,
                            inputs, targets,
                            qtemp, anstemp,
                            num_samples[split_name], 
                            ignore_blanks,
                            natural,
                            pred_tresh, nli_group)
    iterations = num_records["train"]
    val_records = num_records["validation"]
    for logger in [mlog, clog, vlog]:
        logger.info("Iterations:"  + str(iterations))
        logger.info("Val Records:"  + str(val_records))
    warm_up_steps = 0.002*iterations
    #%% tokenizer & model
    if model_id == "test":
        return
    if "mt5" in model_id:
        tokenizer = MT5TokenizerFast.from_pretrained(underlying_model_name)
        model = MT5ForConditionalGeneration.from_pretrained(underlying_model_name)
    elif is_flax:
        tokenizer = AutoTokenizer.from_pretrained(underlying_model_name)
        model = T5ForConditionalGeneration.from_pretrained(underlying_model_name, from_flax=True) 
    else:
        tokenizer = AutoTokenizer.from_pretrained(underlying_model_name)
        model = T5ForConditionalGeneration.from_pretrained(underlying_model_name) 

    allowed_out_token_length = len(tokenizer)
    def clip_logits(logits):
        return logits[:,:,:allowed_out_token_length]
    clip_logits_hook = model.get_output_embeddings().register_forward_hook(
        lambda m,i,o:clip_logits(o)
    )
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

    if do_eval or (not wrap and frozen):
        model.to(device=device)
        mlog.info("Evaluating the model...")
        val_data = atomic_query_responses[val_set]
        eval(model, tokenizer, val_data, inter, save_path, output_name, val_records)  
        return


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
    #%% build dataloader
    train_dataloader = torch.utils.data.DataLoader(atomic_flattened['train'],
        batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn_for_flattened)
    dev_dataloader = torch.utils.data.DataLoader(atomic_flattened['validation'],
        batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn_for_flattened)
    # %% prepare for training
    sw = SummaryWriter(save_path, flush_secs=1)
    tokenizer.save_pretrained(save_path)
    no_decay = ['bias', 'LayerNorm.weight']
    for p in model.parameters():
        p.requires_grad = not frozen 
    if wrap:
        map_relations()
        wrapped_model = wrap_model(model, tokenizer, wrap, load_prompt_path) 
        optimizer_grouped_parameters = [
            {"params":[p for p in wrapped_model.parameters() if p.requires_grad]}
        ]
        wrapped_model.to(device=device)
    else:
        model.to(device=device)
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate,eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,warm_up_steps,iterations)
    step = 0
    best_dev_loss = 1e10
    best_eval_step = 0
    if checkpoint:
        mlog.info("Restoring optimizer and scheduler")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        step = checkpoint['step']
        best_eval_step = checkpoint['best_eval_step']
        best_dev_loss = checkpoint['best_dev_loss']

    #%% tttttt
    train_iter = iter(train_dataloader)
    pbar = tqdm(total=iterations, position=0, leave=True) #,dynamic_ncols=True)
    while step <= iterations and (wrap or not frozen):
        mlog.info("Training...")
        try:
            if (step % cycle == 0 and step > 0): #validation
                with torch.no_grad():
                    if wrap:
                        wrapped_model.update_model_weight()
                    #if frozen:
                    #    for p in model.parameters():
                    #        p.requires_grad = False 
                    model.eval()
                    pbar.set_description('validating...')
                    dev_allset_micro_loss = 0.
                    dev_token_loss = 0.
                    dev_token_count = 0
                    dev_sample_loss = 0. #avg on sample
                    dev_sample_count = 0
                    for batch in tqdm(dev_dataloader,desc=f'validating...',leave=True):
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
                        best_eval_step = step
                        save_checkpoint(model, optimizer, scheduler, step, 
                                        best_eval_step, best_dev_loss,
                                        os.path.join(save_path, "best_model"))

                        generation_results = \
                        "|Queries|Generation Results|\n"\
                        "|-|-|\n"
                        for i,key in enumerate(atomic_flattened['validation']):
                            if i==validation_num_generation:
                                break
                            results = tokenizer.batch_decode(
                                model.generate(**tokenizer(key[0],return_tensors='pt').to(device=device),**generation_params),
                                skip_special_tokens=True
                            )
                            generation_results+=f"|`{key}`|`{str(results)}`|\n"
                        sw.add_text('dev/generation_samples',generation_results,step)
            pbar.set_description('training...')
            pbar.update()
            if unfreez_step > 0 and step > unfreez_step and froze:
                mlog.info("unfreezing the model")
                unfreez_step = 0
                for p in model.parameters():
                    p.requires_grad = True # Unfreezing
            if freez_step > 0 and step > freez_step and not frozen:
                mlog.info("freezing the model")
                freez_step = 0
                for p in model.parameters():
                    p.requires_grad = False # freezing
            model.train()
            optimizer.zero_grad()
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
            batch = {k:v.to(device=device) for k,v in batch.items()}
            if wrap:
                result = wrapped_model(**batch)
            else:
                result = model(**batch)
            loss = result['loss']
            loss.backward()
            optimizer.step()
            scheduler.step()
            step+=1
            sw.add_scalar('train/loss',loss.item(),global_step=step)
            del result
            del loss
        except KeyboardInterrupt:
            mlog.info("exiting while ...")
            break
    # end train while
    pbar.close()
    sw.close()
    if wrap:
        prompt_path = os.path.join(save_path, "prompt")
        Path(prompt_path).mkdir(exist_ok=True, parents=True)
        wrapped_model.prompt_encoder.save(prompt_path)
        with torch.no_grad():
            wrapped_model.update_model_weight()
    save_checkpoint(model, optimizer, scheduler, step, 
                    best_eval_step, best_dev_loss,
                    save_path)

    eval(model, tokenizer, atomic_query_responses[val_set], inter, save_path, output_name, val_records)  

if __name__ == "__main__":
    if "conf_path" in os.environ:
        conf_path=os.environ["conf_path"]
        mlog.info("Reading from conf %s", conf_path)
        if Path(conf_path).exists():
           with open(conf_path, 'r') as f:
               args = json.load(f) 
           print(args)
    else:
        main()
