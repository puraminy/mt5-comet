#%% load libraries
from comet.train.common import *
from comet.train.eval import *
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast, 
    MT5ForConditionalGeneration, MT5TokenizerFast, AdamW, AddedToken,
    GPT2LMHeadModel, GPT2Tokenizer,
    DataCollatorForLanguageModeling,
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
    "--extemp",
    "-et",
    default="",
    type=str,
    help="tempate for examples"
)
@click.option(
    "--method",
    "-mt",
    default="unsup",
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
    "--only_blanks",
    "-ob",
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
    is_flag=True,
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
    "--train_path",
    "-tp",
    default="atomic/train10k.tsv",
    type=str,
    help=""
)
@click.option(
    "--val_path",
    "-vp",
    default="atomic/val.tsv",
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
    default="",
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
    default="5-2",
    type=str,
    help="Encoder-decoder prompt length"
)
@click.option(
    "--prompt_pos",
    "-ppos",
    default="end",
    type=str,
    help=""
)
@click.option(
    "--zero_shot",
    "-zs",
    is_flag=True,
    help=""
)
@click.option(
    "--sampling",
    "-sample",
    default=5,
    type=int,
    help="number of sampels for in context learning"
)
@click.option(
    "--opt_type",
    "-ot",
    default="adam",
    type=str,
    help="optimizer type (adam, ada, ada_no_lr)"
)
@click.option(
    "--samples_per_head",
    "-sph",
    default=0,
    type=int,
    help=""
)
@click.option(
    "--deep_log",
    "-dl",
    is_flag=True,
    help="print more information"
)
@click.option(
    "--trans",
    "-trans",
    default="",
    type=str,
    help=""
)
@click.option(
    "--encoder_type",
    "-et",
    default="lstm",
    type=str,
    help=""
)
@click.option(
    "--from_words",
    "-fw",
    default="",
    type=str,
    help="initialize encoder embeddings from words"
)
@click.option(
    "--rel_filter",
    "-rel",
    default="",
    type=str,
    help=""
)
def train(model_id, experiment, qtemp, anstemp, extemp, method, train_samples, val_set, 
         val_samples, load_path, train_path, val_path, overwrite, save_path, output_name, lang, pred_tresh, ignore_blanks,only_blanks, include, exclude, nli_group, learning_rate, do_eval, inter, cont, wrap, frozen, freez_step, unfreez_step, cpu, load_prompt_path, verbose, cycle, batch_size, path, from_dir, is_flax, config,clear_logs, gen_param, print_log, training_round, epochs_num, is_record, reset_results, start, prompt_length, prompt_pos, zero_shot, sampling, opt_type, samples_per_head, deep_log, trans, encoder_type, from_words,rel_filter):

    #%% some hyper-parameters

    #bbbbbbbbbbb
    #underlying_model_name = "logs/atomic-mt5/last"
    mlog.info("given load path: %s", load_path)
    mlog.info("given save path: %s", save_path)
    if "dlog" in print_log: # data logger
        dlog.addHandler(consoleHandler)
        dlog.setLevel(logging.DEBUG)
    if "vlog" in print_log: # evaluation logger
        vlog.addHandler(consoleHandler)
        vlog.setLevel(logging.DEBUG)
    if "clog" in print_log: # config logger
        clog.addHandler(consoleHandler)
        clog.setLevel(logging.DEBUG)

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
    if not frozen and learning_rate == 0: 
        learning_rate = 6.25e-05 if opt_type == "adam" else 1e-3
        if "gpt" in model_id:
            learning_rate = 1e-5
    if frozen and learning_rate == 0: 
        learning_rate = 0.01  #6.25e-05
    assert learning_rate > 0, "Learning rate is zero!"
    device = 'cuda' if not cpu else 'cpu'
    mlog.info("Optimizer type %s:", opt_type)
    mlog.info("learning rate %s:", learning_rate)

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

    for logger in [mlog, clog, dlog, tlog]:
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        logger.info(f"%%%%%%%%%%%%%%%%%% { model_id } ")
        logger.info(f"%%%%%%%%%%%%%%%%%% { output_name } ")
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    args_str = json.dumps(args, indent=4)
    for logger in [clog]:
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
    #%% load model

    def load_model(model_id, underlying_model_name):
        mlog.info("Loading model ...")
        if model_id == "test":
            return None, None
        elif "gpt" in model_id:
            model = GPT2LMHeadModel.from_pretrained(underlying_model_name)
            tokenizer = AutoTokenizer.from_pretrained(underlying_model_name)
        elif "mt5" in model_id:
            tokenizer = MT5TokenizerFast.from_pretrained(underlying_model_name)
            model = MT5ForConditionalGeneration.from_pretrained(underlying_model_name)
        elif is_flax:
            tokenizer = AutoTokenizer.from_pretrained(underlying_model_name)
            model = T5ForConditionalGeneration.from_pretrained(underlying_model_name, from_flax=True) 
            mlog.info("converting and saving model in %s", save_path)
            tokenizer.save_pretrained(save_path)
            model.save_pretrained(save_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(underlying_model_name)
            model = T5ForConditionalGeneration.from_pretrained(underlying_model_name) 

        return model, tokenizer

    #%% load atomic data
    atomic_dataset = {}
    atomic_dataset["train"] = pd.read_table(train_path)
    atomic_dataset["validation"] = pd.read_table(val_path)
    if trans:
        model, tokenizer = load_model(model_id, underlying_model_name)
        for split_name, df in atomic_dataset.items():
            mlog.info("Translating ...%s ", split_name)
            path = train_path if split_name == "train" else val_path
            model.to(device=device)
            logger = tlog 
            translate(model, tokenizer, df, trans, path, logger, start, load_path) 
        return
    
    model, tokenizer = load_model(model_id, underlying_model_name)
    if from_words and from_words != "rel" and from_words != "none":
        fw_tokens = tokenizer.tokenize(from_words)
        mlog.info("from words ids ***: %s", fw_tokens)
        length = [len(fw_tokens)]
        mlog.info("length got from words ids ***: %s", length)
        set_prompt_lengths(rel_filter, length)
    else:
        length = [int(s) for s in prompt_length.split("-")]
        set_prompt_lengths(rel_filter, length)

    #tokenize_relations(tokenizer)
    atomic_query_responses = {}
    atomic_flattened = {}
    num_records = {}
    num_samples = {"train": train_samples, "validation":val_samples}
    mlog.info("Perparing data ...")
    if model_id in ["t5-large","t5-small", "t5-base", "gpt2"]:
        lang = "en"
    split_lang = {}
    if "-" in lang:
        split_lang["train"] = lang.split("-")[0]
        split_lang["validation"] = lang.split("-")[1]
    else:
        split_lang["train"] = lang
        split_lang["validation"] = lang
    for split_name,split_df in atomic_dataset.items():
        dlog.info("Columns of %s  %s", split_name, "\n".join(list(split_df.columns)))
        dlog.info(split_df.head())
        slang = split_lang[split_name]
        if "2" in slang:
            inp_lang, targ_lang = slang.split("2")
        else:
            inp_lang = targ_lang = slang
        inp_include, inp_exclude = filter_inputs(include, exclude, inp_lang)
        targ_include, targ_exclude = filter_inputs(include, exclude, targ_lang)
        if split_name == "validation":
            samples_per_head = 2
        (atomic_query_responses[split_name], 
         atomic_flattened[split_name],
         num_records[split_name]
        )= fill_data(split_df, split_name,
                            method, prompt_pos, rel_filter,
                            num_samples[split_name], 
                            ignore_blanks,
                            only_blanks,
                            inp_include,
                            inp_exclude,
                            targ_include,
                            targ_exclude,
                            pred_tresh, nli_group, is_record, start, sampling,
                            samples_per_head
                    )
    train_records = num_records["train"]
    val_records = num_records["validation"]
    if deep_log:
        dlog.info(atomic_query_responses["train"])
    for logger in [mlog, clog, vlog]:
        logger.info("Train records:"  + str(train_records))
        logger.info("Val Records:"  + str(val_records))
    if model_id == "test":
        return
    mlog.info("len tokenizer %s", len(tokenizer))
    mlog.info("list of spcial tokens: %s", tokenizer.additional_special_tokens)
    extra = "_" + now
    if experiment == "custom":
        experiment = now
        extra = ""
    results_info = f"{experiment}_{model_id}_{lang}_{method}_{w_str}_{f_str}_tr:{training_round}-ep:{epochs_num}-({start}-{train_records})-{val_records}{extra}"
    if do_eval or (not wrap and frozen):
        mlog.info("Evaluating the model...")
        val_data = atomic_query_responses[val_set]
        model.to(device=device)
        eval(model, tokenizer, val_data, inter, save_path, results_info, val_records, gen_param)  
        return
    accumulation_tiny_steps = 2 
    if "gpt" in model_id:
        accumulation_tiny_steps = 1
    node_batch_size = batch_size//accumulation_tiny_steps
    iterations = train_records//batch_size
    if iterations == 0:
        mlog.info("There is no data to train!!!!!!!!")
        return
    for logger in [mlog, tlog]:
        logger.info("Iterations:"  + str(iterations))
    warm_up_steps = 0.002*iterations
    #%% tokenizer & model
    allowed_out_token_length = len(tokenizer)
    def clip_logits(logits):
        return logits[:,:,:allowed_out_token_length]
    if not "gpt" in model_id:
        clip_logits_hook = model.get_output_embeddings().register_forward_hook(
            lambda m,i,o:clip_logits(o)
        )

    # add new tokens
    # added_tokens = list(atomic_relation_mappings.values()) + [gen_token]
    #%% Prepare training data
    if start > 0 and training_round == 1:
        training_round += 1



# ggggggggg
    attention_mask = None
    def collate_fn_for_flattened(batch):
        global attention_mask
        queries,responses = zip(*batch)
        new_batch = tokenizer(list(queries),return_tensors='pt',padding='longest')
        with tokenizer.as_target_tokenizer():
            tokenized = tokenizer(list(responses),return_tensors='pt',padding='longest')
            labels = tokenized['input_ids']
            labels[labels==tokenizer.pad_token_id] = -100
            new_batch['labels']=labels
            attention_mask = new_batch['attention_mask']
            tlog.info("att mask %s", attention_mask)
            new_batch['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(
                tokenized['input_ids']
            )
            new_batch['decoder_attention_mask'] = tokenized['attention_mask']
        return new_batch
    def collate_fn_for_generation(batch):
         global attention_mask
         queries,responses = zip(*batch)
         inputs = list(queries)
         outputs =list(responses)
         qr = []
         for i in range(len(inputs)):
             qr.append(inputs[i] + " " + outputs[i])

         new_batch = {}
         #tokenized = tokenizer(outputs,return_tensors="pt",
         #        truncation=True,
         #        max_length=256,
         #        padding='max_length')
         tokenized = tokenizer(qr,return_tensors='pt',padding='longest')
         labels = tokenized['input_ids'].detach().clone()
         labels[labels==tokenizer.pad_token_id] = -100
         new_batch['input_ids']=tokenized['input_ids']
         new_batch['attention_mask']=tokenized['attention_mask']
         attention_mask = new_batch['attention_mask']
         new_batch['labels']=labels
         return new_batch #,references
    #%% build dataloader
    if "gpt" in model_id: 
        tokenizer.add_special_tokens(pad_token)
        tokenizer.add_special_tokens(sep_token)
        mlog.info("pad token id: %s", tokenizer.pad_token_id)
        data_collator = collate_fn_for_generation
    else:
        data_collator = collate_fn_for_flattened

    train_dataloader = torch.utils.data.DataLoader(atomic_flattened['train'],
        batch_size=node_batch_size,shuffle=shuffle,collate_fn=data_collator)
    dev_dataloader = torch.utils.data.DataLoader(atomic_flattened['validation'],
        batch_size=node_batch_size,shuffle=shuffle,collate_fn=data_collator)
    # %% prepare for training
    sw = SummaryWriter(save_path, flush_secs=1)
    no_decay = ['bias', 'LayerNorm.weight']
    if frozen:
        for p in model.parameters():
            p.requires_grad = False 
    else:
        for p in model.parameters():
            p.requires_grad = True 
    wrapped_model = None
    if wrap:
        if not load_prompt_path and Path(os.path.join(load_path, model_id, "prompt")).exists():
            load_prompt_path = os.path.join(load_path, model_id, "prompt")
            mlog.info("prompt path:%s ", load_prompt_path)
        mlog.info("Wrapping the model ...")
        wrapped_model = wrap_model(model, tokenizer, rel_filter, encoder_type, load_prompt_path, from_words = from_words) 
    if wrapped_model:
        wrapped_model.to(device=device)
        mlog.info("len tokenizer after wrapping %s", len(tokenizer))
    else:
        wrap = False
        extend_tokenizer(tokenizer, "")
        mlog.info("len tokenizer after extending %s", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))
        model.to(device=device)


    tokenizer.save_pretrained(save_path)
    def get_optimizer(model, learning_rate, wrap, opt_type):
        if wrap:
            optimizer_grouped_parameters = [
                {"params":[p for p in wrapped_model.parameters() if p.requires_grad]}
            ]
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if opt_type == "adam":
            optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate,eps=1e-8)
            scheduler = get_linear_schedule_with_warmup(optimizer,warm_up_steps,iterations)
        elif opt_type == "ada_no_lr":
            optimizer = Adafactor(optimizer_grouped_parameters, 
                    scale_parameter=True, 
                    relative_step=True, warmup_init=True, lr=None)
            scheduler = AdafactorSchedule(optimizer)
        elif opt_type == "ada":
            # replace AdamW with Adafactor
            optimizer = Adafactor(
                model.parameters(),
                lr=learning_rate,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=0.0,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False
            )
            scheduler = AdafactorSchedule(optimizer)
        else:
            raise ValueError(opt_type + " must be one of adam, ada, ada_no_lr")
        return optimizer, scheduler

    optimizer, scheduler = get_optimizer(model, learning_rate, wrap, opt_type) 
    step = 0
    best_dev_loss = 100
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
    if epochs_num == 0 or not wrap and frozen:
        mlog.info("Skip training...")
    elif step <= iterations and (wrap or not frozen):
        mlog.info("Training...")
    pbar = tqdm(total=iterations, position=0, leave=True) #,dynamic_ncols=True)
    for epoch in range(epochs_num):
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
                            tlog.info("epoch %s, best_eval_step: %s", epoch, best_eval_step)
                            #save_checkpoint(model, optimizer, scheduler, step, 
                            #                best_eval_step, best_dev_loss,
                            #                os.path.join(save_path, "best_model"))

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
                    optimizer, scheduler = get_optimizer(model, last_lr, wrap, opt_type)
                if freez_step > 0 and step > freez_step and not frozen:
                    mlog.info("freezing the model")
                    freez_step = 0
                    for p in model.parameters():
                        p.requires_grad = False # freezing
                    last_lr = scheduler.get_last_lr()[0]
                    optimizer, scheduler = get_optimizer(model, last_lr, wrap, opt_type)
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
                    if "loss" in result:
                        loss = result['loss']/accumulation_tiny_steps
                    else:
                        loss = torch.nn.functional.cross_entropy(
                            result['logits'].reshape(-1,result['logits'].size(2)),
                            batch['labels'].reshape(-1,),
                            reduction='none'
                        ).reshape(result['logits'].size(0),-1)
                        #loss /= accumulation_tiny_steps
                        loss = loss.mean()
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
                pbar.set_description(f'training ...[loss:{bloss:.2f} ({mean_loss:.2f}) best:{best_eval_step} {best_dev_loss:.2f}]')
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
        if wrapped_model.prompt_encoder:
            prompt_path = os.path.join(save_path, "prompt", "encoder")
            Path(prompt_path).mkdir(exist_ok=True, parents=True)
            mlog.info("Saving encoder prompt at %s", prompt_path)
            wrapped_model.prompt_encoder.save(prompt_path)
        if wrapped_model.decoder_prompt_encoder:
            prompt_path = os.path.join(save_path, "prompt", "decoder")
            Path(prompt_path).mkdir(exist_ok=True, parents=True)
            mlog.info("Saving decoder prompt at %s", prompt_path)
            wrapped_model.decoder_prompt_encoder.save(prompt_path)
        with torch.no_grad():
            mlog.info("Updating the model weights before evaluaton...")
            wrapped_model.update_model_weight()
    save_checkpoint(model, optimizer, scheduler, step, 
                    best_eval_step, best_dev_loss,
                    save_path)

    eval(model, tokenizer, atomic_query_responses[val_set], inter, save_path, results_info, val_records, gen_param, attention_mask)  

#ettt

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


def translate(model, tokenizer, df, trans_col, path, logger=None, start=0, save_path=""):
    pbar = tqdm(total= len(df))
    oldcol, newcol,save_step = trans_col.split("@")
    newcol = oldcol + "_" + newcol
    save_step = int(save_step)
    trans = []
    mlog.info("len(df): %s", len(df))
    mlog.info("save_path: %s", save_path)
    fname = Path(path).stem
    ii = 0
    for idx, row in df.iterrows():
        if ii < start:
            ii += 1
            continue
        try:
            hyps = gen_resp(model, tokenizer, row[oldcol])
        except Exception as e:
            mlog.info("Error: %s:", e)
            continue
        _t = hyps[0]
        trans_row = {newcol:_t, oldcol:row[oldcol], "prefix":row["prefix"], "input_text":row["input_text"]}
        trans.append(trans_row)
        if len(trans) < 5:
            mlog.inf("row: %s", trans_row)
        pbar.update()
        if len(trans) % save_step == 0 or len(trans) == 5:
            p = os.path.join(save_path, fname + str(ii).replace("000","k_") + ".tsv")
            if logger:
                logger.info("Saving at %s", p)
                logger.info("Len trans: %s", len(trans))
            new_df = pd.DataFrame(data=trans) 
            trans = []
            new_df.to_csv(p, sep="\t", index=False)
        ii += 1

    new_df = pd.DataFrame(data=trans) 
    p = os.path.join(save_path, fname + str(ii).replace("000","k_") + ".tsv")
    df.to_csv(p, sep="\t", index=False)



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
    df = pd.DataFrame(sd, columns=["exp","model","lang", "method","wrap","frozen","epochs","stype", "dir", "score"])
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
