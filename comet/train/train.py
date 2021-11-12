#%% load libraries
from comet.train.common import *
from comet.train.eval import *
from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast, 
    MT5ForConditionalGeneration, MT5TokenizerFast, AdamW, AddedToken,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
import torch
import re
import json
import glob
from torch.utils.tensorboard import SummaryWriter
import os,time
import argparse
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import click
from tqdm import tqdm


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Run    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@click.group(invoke_without_command=True)
@click.option(
    "--conf_path",
    "-cp",
    default="confs",
    type=str,
    help=""
)
@click.option(
    "--experiment",
    "-exp",
    default="conf",
    type=str,
    help="Select the pattern of configurations files for an experiment (starts with)"
)
@click.option(
    "--print_log",
    "-p",
    default="mlog",
    type=str,
    help=""
)
@click.option(
    "--model_id",
    "-m",
    default="",
    type=str,
    help=""
)
@click.option(
    "--train_samples",
    "-n",
    default=0,
    type=int,
    help=""
)
@click.option(
    "--recal",
    "-rc",
    is_flag=True,
    help=""
)
@click.option(
    "--exclude",
    "-ex",
    default="",
    type=str,
    help=""
)
@click.option(
    "--reset_results",
    "-reset",
    is_flag=True,
    help=""
)
@click.pass_context
#rrrrrrrrrrr
def run(ctx, conf_path, experiment, print_log, model_id, train_samples, recal, 
        exclude, reset_results):
     global results
     if ctx.invoked_subcommand is None:
        if reset_results:
            set_results({})
        mlog.info("Reading from conf %s", conf_path)
        confs = glob.glob(f"{conf_path}/*")
        default_model = ""
        for conf in confs:
            fname = Path(conf).stem
            mlog.info(f"%%% {fname} %%%")
            if not experiment in fname:
                mlog.info("Skipping .... This was not in experiments")
                continue
            if exclude and exclude in fname:
                mlog.info("Skipping .... by exclude")
                continue
            val = getVal(fname, results) 
            mlog.info("current val: {}".format(val))
            if val != "NA" and not recal:
                mlog.info("Skipping .... This was done before")
                continue
            if Path(conf).exists():
               with open(conf, 'r') as f:
                   args = json.load(f) 
               args["print_log"] = print_log
               if train_samples > 0:
                   args["train_samples"] = train_samples
               if model_id:
                   if default_model and args["model_id"] != default_model:
                       break
                   elif args["model_id"] != default_model:
                       default_model = args["model_id"]
                   mlog.info(f"Replacing {default_model} with {model_id}")
                   args["model_id"] = model_id
                   out = args["output_name"].split("_")
                   out[1] = model_id
                   args["output_name"] = "_".join(out)
               ctx.invoke(train, **args)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@run.command()
@click.argument("model_id", type=str)
@click.option(
    "--experiment",
    "-exp",
    default="custom",
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
    default=40,
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
    default="",
    type=str,
    help=""
)
@click.option(
    "--overwrite",
    "-ow",
    default="",
    type=str,
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
    default="mix",
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
    default="{ph} {resp} {end}",
    type=str,
    help="tempate for response"
)
@click.option(
    "--method",
    "-mt",
    default="",
    type=str,
    help="Based on the method (sup, unsup, context-en, ... ) templates for query and answer are created."
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
    "--include",
    "-inc",
    default="",
    type=str,
    help="filter input columns (must have this substring)"
)
@click.option(
    "--exclude",
    "-exc",
    default="",
    type=str,
    help="filter target columns (must have this substring)"
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
    default=0,
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
@click.option(
    "--config",
    "-cfg",
    is_flag=True,
    help="Only create a configuration file from input parameters"
)
@click.option(
    "--clear_logs",
    "-cl",
    is_flag=True,
    help=""
)
@click.option(
    "--gen_param",
    "-gp",
    default="greedy",
    type=str,
    help=""
)
@click.option(
    "--print_log",
    "-print",
    default="mlog",
    type=str,
    help=""
)
@click.option(
    "--epochs_num",
    "-eps",
    default=1,
    type=int,
    help=""
)
@click.option(
    "--training_round",
    "-tr",
    default=1,
    type=int,
    help="If you want to retrain a model or continue trainig it on new data set it incremntally"
)
@click.option(
    "--is_record",
    "-recs",
    is_flag=True,
    help="Show if train_samples are records or unique heads"
)
@click.option(
    "--reset_results",
    "-reset",
    is_flag=True,
    help=""
)
@click.option(
    "--start",
    "-start",
    default=0,
    type=int,
    help="Start record number for training data"
)
@click.option(
    "--prompt_length",
    "-pl",
    default="5-0",
    type=str,
    help="Encoder-decoder prompt length"
)
@click.option(
    "--prompt_pos",
    "-ppos",
    default="start",
    type=str,
    help=""
)
@click.option(
    "--zero_shot",
    "-zs",
    is_flag=True,
    help=""
)
def train(model_id, experiment, qtemp, anstemp, method, train_samples, val_set, 
         val_samples, load_path, overwrite, save_path, output_name, lang, pred_tresh, ignore_blanks, include, exclude, nli_group, learning_rate, do_eval, inter, cont, wrap, frozen, freez_step, unfreez_step, cpu, load_prompt_path, verbose, cycle, batch_size, path, from_dir, is_flax, config,clear_logs, gen_param, print_log, training_round, epochs_num, is_record, reset_results, start, prompt_length, prompt_pos, zero_shot):

    #%% some hyper-parameters

    #bbbbbbbbbbb
    #underlying_model_name = "logs/atomic-mt5/last"
    mlog.info("given load path: %s", load_path)
    mlog.info("given save path: %s", save_path)
    if "dlog" in print_log: # data logger
        dlog.addHandler(consoleHandler)
    if "vlog" in print_log: # evaluation logger
        vlog.addHandler(consoleHandler)
    if "clog" in print_log: # config logger
        clog.addHandler(consoleHandler)
    if method:    
        qtemp, anstemp = create_templates(method, wrap, frozen,
                gen_pos="end", prompt_pos=prompt_pos, zero_shot=zero_shot, lang=lang)
    if lang:
        include, exclude = filter_inputs(include, exclude, lang)

    args = locals() # input parameters

    mlog.info("========================= Version 7 ========================")
    if save_path == "":
        if "ahmad" or "pouramini" in home:
            save_path = os.path.join(home, "logs")
        else:
            save_path = "/content/drive/MyDrive/pouramini/logs"

    w_str = "wrapped" if wrap else "unwrapped"
    f_str = "freezed" if frozen else "unfreezed"
    if not output_name:
        output_name = f"{experiment}_{model_id}-{train_samples}_{lang}_{method}_{w_str}_{f_str}"
    conf_path = os.path.join(save_path,"confs")
    if model_id == "test":
        save_path = ""
        output_name = "test"
        conf_path = os.path.join(home, "logs/confs")
    Path(conf_path).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(conf_path, f'conf_{output_name}.json'), 'w') as outfile:
        json.dump(args, outfile, indent=4)



    if config:
        mlog.info("Config %s was created at %s", "conf_" + output_name, conf_path)
        return

    if save_path != logPath:
        for logger, fname in zip([mlog,dlog,clog,vlog,tlog], ["main","data","cfg","eval","train"]):
            if len(logger.handlers) >= 2:
                continue
            logger.setLevel(logging.INFO)
            logFilename = os.path.join(save_path, fname + ".log")
            handler = logging.FileHandler(logFilename, mode = "w" if clear_logs else "a")
            logger.addHandler(handler)

    if not load_path:
        load_path = os.path.join(home, "pret")

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
    device = 'cuda' if not cpu else 'cpu'

    log_dir = save_path
    set_device(device)
    output_name = model_id if not output_name else output_name
    save_path = os.path.join(log_dir, output_name)
    model_name = f"{learning_rate}_{cycle}_{train_samples}"
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

    for logger in [mlog, vlog, clog, dlog, tlog]:
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        logger.info(f"%%%%%%%%%%%%%%%%%% { model_id } ")
        logger.info(f"%%%%%%%%%%%%%%%%%% { output_name } ")
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    args_str = json.dumps(args, indent=4)
    for logger in [clog, vlog]:
        logger.info(args_str)
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    do_overwrite = False
    if overwrite:
        save_path = os.path.join(log_dir, overwrite)
        do_overwrite = True
    ii = 1
    while not do_overwrite and Path(save_path).exists() and not model_id=="test":
        ans = input(f"The output directory {save_path} already exists, do you want to overwrite it? (y/n)")
        if ans == "y":
            do_overwrite = True 
            break
        save_path = os.path.join(log_dir,output_name + "_"+str(ii))
        mlog.info(save_path)
        ii += 1

    mlog.info(f"SAVE Path:{save_path}" + " (Overwrite)" if overwrite else "")
    mlog.info(f"LOAD Path:{underlying_model_name}")
    Path(save_path).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(save_path, "best_model")).mkdir(exist_ok=True, parents=True)
    #%% load atomic data
    import pandas as pd
    atomic_dataset = {}
    train_path= "atomic/xIntent_en_fa_train_no_dups.tsv"
    val_path= "atomic/xIntent_en_fa_validation_no_dups.tsv"
    atomic_dataset["train"] = pd.read_table(train_path)
    atomic_dataset["validation"] = pd.read_table(val_path)

    length = prompt_length.split("-")
    enc_pl = int(length[0]) 
    dec_pl = int(length[1])
    map_relations_to_prompts(wrap, enc_pl, dec_pl)
    atomic_query_responses = {}
    atomic_flattened = {}
    num_records = {}
    num_samples = {"train": train_samples, "validation":val_samples}
    for split_name,split_df in atomic_dataset.items():
        (atomic_query_responses[split_name], 
         atomic_flattened[split_name],
         num_records[split_name]
        )= fill_data(split_df, split_name,
                            qtemp, anstemp,
                            num_samples[split_name], 
                            ignore_blanks,
                            include,
                            exclude,
                            pred_tresh, nli_group, is_record, start)
    train_records = num_records["train"]
    val_records = num_records["validation"]
    for logger in [mlog, clog, vlog]:
        logger.info("Train records:"  + str(train_records))
        logger.info("Val Records:"  + str(val_records))
    accumulation_tiny_steps = 2 
    node_batch_size = batch_size//accumulation_tiny_steps
    iterations = train_records//batch_size
    for logger in [mlog, tlog]:
        logger.info("Iterations:"  + str(iterations))
    warm_up_steps = 0.002*iterations
    #%% tokenizer & model
    mlog.info("Loading model ...")
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
    mlog.info("len tokenizer %s", len(tokenizer))
    extend_tokenizer(tokenizer, "")
    mlog.info("len tokenizer after extending %s", len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    #%% Prepare training data
    if start > 0 and training_round == 1:
        training_round += 1
    extra = "_" + now
    if experiment == "custom":
        experiment = now
        extra = ""
    results_info = f"{experiment}_{model_id}_{lang}_{method}_{w_str}_{f_str}_tr:{training_round}-ep:{epochs_num}-({start}-{train_records})-{val_records}{extra}"

    if do_eval or (not wrap and frozen):
        model.to(device=device)
        mlog.info("Evaluating the model...")
        val_data = atomic_query_responses[val_set]
        eval(model, tokenizer, val_data, inter, save_path, results_info, val_records, gen_param)  
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
        batch_size=node_batch_size,shuffle=shuffle,collate_fn=collate_fn_for_flattened)
    dev_dataloader = torch.utils.data.DataLoader(atomic_flattened['validation'],
        batch_size=node_batch_size,shuffle=shuffle,collate_fn=collate_fn_for_flattened)
    # %% prepare for training
    sw = SummaryWriter(save_path, flush_secs=1)
    tokenizer.save_pretrained(save_path)
    no_decay = ['bias', 'LayerNorm.weight']
    if frozen:
        for p in model.parameters():
            p.requires_grad = False 
    else:
        for p in model.parameters():
            p.requires_grad = True 
    if wrap:
        if not load_prompt_path and Path(os.path.join(load_path, model_id, "prompt")).exists():
            load_prompt_path = os.path.join(load_path, model_id, "prompt")
            mlog.info("prompt path:%s ", load_prompt_path)
        mlog.info("Wrapping the model ...")
        wrapped_model = wrap_model(model, tokenizer, wrap, False, load_prompt_path) 
        wrapped_model.to(device=device)
    else:
        model.to(device=device)

    def get_optimizer(model, learning_rate, wrap):
        if wrap:
            optimizer_grouped_parameters = [
                {"params":[p for p in wrapped_model.parameters() if p.requires_grad]}
            ]
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate,eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,warm_up_steps,iterations)
        return optimizer, scheduler

    optimizer, scheduler = get_optimizer(model, learning_rate, wrap) 
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
    mlog.info("batch size: %s", batch_size)
    mlog.info("node batch size: %s", node_batch_size)
    if step <= iterations and (wrap or not frozen):
        mlog.info("Training...")
    for epoch in range(epochs_num):
        pbar = tqdm(total=iterations, position=0, leave=True) #,dynamic_ncols=True)
        mlog.info(f"============== epoch {epoch}\n")
        tlog.info(f"============== epoch {epoch}\n")
        tot_loss = 0
        step = 0
        train_iter = iter(train_dataloader)
        while step < iterations-1 and (wrap or not frozen):
            try:
                if cycle > 0 and (step % cycle == 0 and step > 0): #validation
                    with torch.no_grad():
                        if wrap:
                            wrapped_model.update_model_weight()
                        model.eval()
                        pbar.set_description(f'validating on {cycle}...')
                        vlog.info(f'validating on {cycle}...')
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
                        vlog.info('dev/micro_avg_loss: %s-%s',dev_micro_avg_loss,step)
                        sw.add_scalar('dev/macro_avg_loss',dev_macro_avg_loss,step)
                        vlog.info('dev/macro_avg_loss: %s-%s',dev_macro_avg_loss,step)
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
                                results = gen_resp(model, tokenizer, key[0]) 
                                vlog.info(results)
                                generation_results+=f"|`{key}`|`{str(results)}`|\n"
                            sw.add_text('dev/generation_samples',generation_results,step)
                if unfreez_step > 0 and step > unfreez_step and froze:
                    mlog.info("unfreezing the model")
                    unfreez_step = 0
                    for p in model.parameters():
                        p.requires_grad = True # Unfreezing
                    last_lr = scheduler.get_last_lr()[0]
                    optimizer, scheduler = get_optimizer(model, last_lr, wrap)
                if freez_step > 0 and step > freez_step and not frozen:
                    mlog.info("freezing the model")
                    freez_step = 0
                    for p in model.parameters():
                        p.requires_grad = False # freezing
                    last_lr = scheduler.get_last_lr()[0]
                    optimizer, scheduler = get_optimizer(model, last_lr, wrap)
                model.train()
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.)
                for tiny_step in range(accumulation_tiny_steps):
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        tlog.info("Stop Iteration occured at %s", step)
                        train_iter = iter(train_dataloader)
                        batch = next(train_iter)
                    batch = {k:v.to(device=device) for k,v in batch.items()}
                    if wrap:
                        result = wrapped_model(**batch)
                    else:
                        result = model(**batch)
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
                bloss = batch_loss.item()
                tot_loss += bloss
                mean_loss = tot_loss/step
                sw.add_scalar('train/loss',bloss,global_step=step)
                tlog.info("{:<5}: {:6.2f} > {:6.2f}".format(step, bloss, mean_loss))
                pbar.set_description(f'training ...[loss:{bloss:.2f} ({mean_loss:.2f})]')
                pbar.update()
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
        mlog.info("Saving prompt at %s", prompt_path)
        wrapped_model.prompt_encoder.save(prompt_path)
        with torch.no_grad():
            wrapped_model.update_model_weight()
    save_checkpoint(model, optimizer, scheduler, step, 
                    best_eval_step, best_dev_loss,
                    save_path)

    eval(model, tokenizer, atomic_query_responses[val_set], inter, save_path, results_info, val_records, gen_param)  

@run.command()
@click.argument("experiment", type=str)
@click.option(
    "--models_dir",
    "-m",
    default="/content/drive/MyDrive",
    type=str,
    help=""
)
def create_confs(experiment, models_dir):
    #cccccccccccc
    print("Creating configurations...")
    base_dir = home
    conf = os.path.join(base_dir, "logs/confs/conf_test.json")
    save_path = os.path.join(base_dir, "mt5-comet/comet/train/")
    conf_path = os.path.join(save_path,"confs")
    Path(conf_path).mkdir(exist_ok=True, parents=True)
    if Path(conf).exists():
       with open(conf, 'r') as f:
           args = json.load(f) 
    else:
        print(conf + " doesn't exists!")
        return
    samples = 300
    args["experiment"] = experiment
    args["train_samples"] = samples
    args["val_samples"] = 50
    args["cycle"] = 0
    args["load_path"] = os.path.join(models_dir, "pret")
    args["save_path"] = os.path.join(models_dir, "logs")
    args["overwrite"] = experiment
    args["cpu"] = False 
    args["config"] = False 
    args["batch_size"] = 2 
    args["gen_param"] = "greedy" 
    args["exclude"] = "natural" 
    models = {"fat5-large":True, "fat5-3k-gen":True}
    langs = {"en":True, "fa":True, "mix":True}
    methods = {"unsup":True, "context-en":True,"context-fa":False, "sup": True}
    to_wrap = {"wrapped":True, "unwrapped": True}
    to_freez = {"freezed":True, "unfreezed": True}
    ii = 0
    for model in [k for k in models.keys() if models[k]]:
        for lang in [k for k in langs.keys() if langs[k]]: 
            for method in [k for k in methods.keys() if methods[k]]:
                for w in [k for k in to_wrap.keys() if to_wrap[k]]:
                   for f in [k for k in to_freez.keys() if to_freez[k]]:
                       if method == "context-en" and lang == "en":
                           continue
                       if method == "context-fa" and lang == "fa":
                           continue
                       if f == "freezed" and method == "sup" and w == "unwrapped":
                           continue
                       args["model_id"]= model
                       args["frozen"] = False
                       if f == "freezed":
                           args["frozen"] = True
                       args["wrap"] = ""
                       if w == "wrapped":
                           args["wrap"] = "xIntent"
                       name = f"{experiment}_{model}-{samples}_{lang}_{method}_{w}_{f}"
                       args["output_name"] = name
                       #name = name.replace("_unwrapped", "")
                       #name = name.replace("_unfreezed", "")
                       ii +=1
                       print(str(ii) + ":" + name)
                       with open(os.path.join(conf_path, f'{name}.json'), 'w') as outfile:
                                json.dump(args, outfile, indent=4)

@run.command()
@click.option(
    "--stype",
    "-s",
    default="rouge",
    type=str,
    help="score type (rouge, bert, etc.)"
)
@click.option(
    "--model",
    "-m",
    default="",
    type=str,
    help=""
)
@click.option(
    "--method",
    "-mt",
    default="",
    type=str,
    help=""
)
@click.option(
    "--sort",
    "-so",
    default="score",
    type=str,
    help=""
)
def res(stype, model, method, sort):
    mlog.info("Reading results from %s", resPath)
    with open(os.path.join(resPath, "results.json"), "r") as f:
        data = json.load(f)
    
    sd = superitems(data)
    df = pd.DataFrame(sd, columns=["exp","model","lang", "method","wrap","frozen","stype", "dir", "score"])
    df.to_csv(os.path.join(resPath, "table_all.tsv"), sep="\t", index = False)
    if stype == "all":
        print(df)
        return
    df = df[df["stype"] == stype]
    del df["stype"] 
    if sort:
        df = df.sort_values(by=[sort], ascending=False)
    df.to_csv(os.path.join(resPath, f"table_{stype}.tsv"), sep="\t", index = False)
    print(df)

if __name__ == "__main__":
   run()
