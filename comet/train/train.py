#%% load libraries
from comet.train.common import *
import itertools, collections
import shutil
from comet.train.eval import *
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast, 
    AutoModelForSeq2SeqLM, 
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
    default="",
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
    "--overwrite",
    "-ow",
    is_flag=True,
    help=""
)
@click.pass_context
#rrrrrrrrrrr
def run(ctx, conf_path, experiment, print_log, model_id, train_samples, recal, 
        exclude, overwrite):
     if not conf_path:
        conf_path = "confs"
        if colab: conf_path = "colab_confs"
     if ctx.invoked_subcommand is None:
        mlog.info("Reading from conf %s", conf_path)
        confs = sorted(glob.glob(f"{conf_path}/*"))
        default_model = ""
        first = True
        for conf in confs:
            fname = Path(conf).stem
            mlog.info(f"%%% {fname} %%%")
            if not experiment in fname:
                mlog.info("Skipping .... This was not in experiments")
                continue
            if exclude and exclude in fname:
                mlog.info("Skipping .... by exclude")
                continue
            if Path(conf).exists():
               with open(conf, 'r') as f:
                   args = json.load(f) 
               args["print_log"] = print_log
               spath = args["save_path"]
               if first and Path(spath).exists():
                   mlog.info("%s already exists!", spath)
                   first = False
                   if overwrite:
                      shutil.rmtree(spath)
                   else:
                      os.rename(spath, spath + "_backup_" + now)

               Path(spath).mkdir(exist_ok=True, parents=True)
               #mlog.info("save path: %s", spath)
               cur_res_path = os.path.join(spath, args["output_name"], "new_result*")
               #mlog.info("cur_res_path: %s", cur_res_path)
               cur_res = glob.glob(cur_res_path)
               #mlog.info("cur_res: %s", cur_res)
               if cur_res and not recal:
                    mlog.info("Skipping .... This was done before %s ", spath)
                    continue
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
                   if args["load_path"]:
                       shutil.copy(conf, args["load_path"])
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
    default=0,
    type=int,
    help=""
)
@click.option(
    "--test_set",
    "-ts",
    default="test",
    type=str,
    help=""
)
@click.option(
    "--val_samples",
    "-vn",
    default=150,
    type=int,
    help=""
)
@click.option(
    "--test_samples",
    "-tn",
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
    default="en",
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
    default="atomic/train.tsv",
    type=str,
    help=""
)
@click.option(
    "--val_path",
    "-vp",
    default="atomic/val_all_rels.tsv",
    type=str,
    help=""
)
@click.option(
    "--test_path",
    "-tep",
    default="atomic/test.tsv",
    type=str,
    help=""
)
@click.option(
    "--sample_path",
    "-sp",
    default="atomic/val_all_rels.tsv",
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
    "--per_record",
    "-recs",
    is_flag=True,
    help="Show if train_samples are records or unique heads"
)
@click.option(
    "--is_even",
    "-even",
    is_flag=True,
    help="Show if data set has equal number of records for each relation" 
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
    default="",
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
@click.option(
    "--ex_type",
    "-ext",
    default="",
    type=str,
    help=""
)
@click.option(
    "--last_data",
    "-last",
    is_flag=True,
    help=""
)
@click.option(
    "--save_df",
    "-sdf",
    default="",
    type=str,
    help=""
)
@click.option(
    "--merge_prompts",
    "-mp",
    is_flag=True,
    help=""
)
@click.option(
    "--num_workers",
    "-nw",
    default=0,
    type=int,
    help=""
)
@click.option(
    "--no_score",
    "-nos",
    is_flag=True,
    help=""
)
@click.option(
    "--train_start",
    "-tstart",
    default=0,
    type=int,
    help=""
)
@click.option(
    "--no_save_model",
    "-nsm",
    is_flag=True,
    help=""
)
def train(model_id, experiment, qtemp, anstemp, extemp, method, train_samples, test_set, 
         val_samples, test_samples, load_path, train_path, val_path, test_path, sample_path, overwrite, save_path, output_name, lang, pred_tresh, ignore_blanks,only_blanks, include, exclude, nli_group, learning_rate, do_eval, cont, wrap, frozen, freez_step, unfreez_step, cpu, load_prompt_path, verbose, cycle, batch_size, path, from_dir, is_flax, config,clear_logs, gen_param, print_log, training_round, epochs_num, per_record, is_even, start, prompt_length, prompt_pos, zero_shot, sampling, opt_type, samples_per_head, deep_log, trans, encoder_type, from_words,rel_filter, ex_type, last_data, save_df, merge_prompts, num_workers, no_score, train_start, no_save_model):

    #%% some hyper-parameters

    #bbbbbbbbbbb
    #underlying_model_name = "logs/atomic-mt5/last"
    mlog.info("given load path: %s", load_path)
    vlog.info("given load path: %s", load_path)
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
    if not output_name and not (cont or do_eval):
        output_name = method
    conf_path = save_path
    if model_id == "test":
        save_path = ""
        output_name = "test"
        conf_path = os.path.join(home, "logs/confs")
    Path(conf_path).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(conf_path, f'exp_conf.json'), 'w') as outfile:
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
    validation_size = val_samples 
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
    mlog.info("Output name: %s", output_name)
    save_path = os.path.join(log_dir, output_name)
    model_name = f"{learning_rate}_{cycle}_{train_samples}"
    checkpoint = None
    if not Path(underlying_model_name).exists() and Path(save_path).exists() and not model_id=="test" and (cont or do_eval):
        mlog.info("Loading from %s", save_path)
        underlying_model_name = save_path
        checkpoint_path = os.path.join(save_path,"saved_states")
        if cont:
            if Path(checkpoint_path).exists():
                checkpoint = torch.load(checkpoint_path)
            if Path(conf_path).is_file():
               with open(conf_path, 'r') as f:
                   args = json.load(f) 
               mlog.info(args)
               mlog.info("Loading from configuration file")
               qtemp = args['qtemp']
               atemp = args['anstemp']
               mlog.info("Qtemp: %s", args['qtemp'])
               mlog.info("Anstemp: %s", args['anstemp'])

    for logger in [mlog, clog, dlog, tlog, vlog]:
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        logger.info(f"%%%%%%%%%%%%%%%%%% { model_id } ")
        logger.info(f"%%%%%%%%%%%%%%%%%% { output_name } ")
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    args_str = json.dumps(args, indent=4)
    for logger in [clog]:
        logger.info(args_str)
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    do_overwrite = False
    if overwrite or do_eval:
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
            tokenizer = AutoTokenizer.from_pretrained(underlying_model_name, add_prefix_space=True)
        elif "mt5" in model_id:
            tokenizer = MT5TokenizerFast.from_pretrained(underlying_model_name)
            model = MT5ForConditionalGeneration.from_pretrained(underlying_model_name)
        elif is_flax:
            tokenizer = AutoTokenizer.from_pretrained(underlying_model_name)
            model = T5ForConditionalGeneration.from_pretrained(underlying_model_name, from_flax=True) 
            mlog.info("converting and saving model in %s", save_path)
            tokenizer.save_pretrained(save_path)
            model.save_pretrained(save_path)
        elif "bart" in model_id.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(underlying_model_name)
            tokenizer = AutoTokenizer.from_pretrained(underlying_model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(underlying_model_name)
            model = T5ForConditionalGeneration.from_pretrained(underlying_model_name) 

        if underlying_model_name == model_id:
            mlog.info("Saving model on local %s", load_path)
            model.save_pretrained(os.path.join(load_path, model_id))
            tokenizer.save_pretrained(os.path.join(load_path, model_id))
            underlying_model_name = os.path.join(load_path, model_id)
        return model, tokenizer, underlying_model_name

    #%% load atomic data
    atomic_dataset = {}
    if not save_df:
        if not "@" in ex_type:
            save_df = "all_rels"
        else:
            _, save_df = ex_type.split("@")
    if False: #save_df:
        _train_path = train_path.replace(".tsv", "_" + save_df + ".tsv")
        _val_path = val_path.replace(".tsv",  "_" + save_df + ".tsv")
        if Path(_train_path).exists():
            train_path = _train_path
            mlog.info("Loading train data...")
        if Path(_val_path).exists():
            val_path = _val_path
            mlog.info("Loading val data...")
    if trans:
        model, tokenizer, underlying_model_name = load_model(model_id, underlying_model_name)
        for split_name, df in atomic_dataset.items():
            mlog.info("Translating ...%s ", split_name)
            if trans != split_name:
                continue
            path = train_path if split_name == "train" else val_path
            model.to(device=device)
            logger = tlog 
            mlog.info("Translating %s", path)
            trans_df = translate(model, tokenizer, df, "target_text@fa@5000", path, logger, start, load_path) 
            translate(model, tokenizer, trans_df, "input_text@fa@5000", path, logger, start, load_path) 
        return
    
    model, tokenizer, underlying_model_name = load_model(model_id, underlying_model_name)
    if from_words and from_words != "rel" and from_words != "none":
        fw_tokens = tokenizer.tokenize(from_words)
        mlog.info("from words ids ***: %s", fw_tokens)
        length = [len(fw_tokens)]
        mlog.info("length got from words ids ***: %s", length)
        set_prompt_lengths(rel_filter, length)
    elif prompt_length:
        length = [int(s) for s in prompt_length.split("-")]
        set_prompt_lengths(rel_filter, length)

    num_samples = {"train": train_samples, "validation":val_samples, "sample":0, "test":test_samples}
    split_path = {"train":train_path, "validation":val_path, "sample":sample_path, "test":test_path}
    save_ds_path = {}
    for split, _path in split_path.items():
        #_path = _path.replace(".tsv","_")
        save_ds_path[split] = os.path.join(underlying_model_name, split)
    #tokenize_relations(tokenizer)
    atomic_query_responses = {"train":[], "validation":[]}
    generate_samples = {}
    mlog.info("Perparing data ...")
    if model_id in ["t5-large","t5-small", "t5-base", "gpt2"]:
        lang = "en"
    split_lang = {}
    if "-" in lang:
        split_lang["train"] = lang.split("-")[0]
        split_lang["sample"] = lang.split("-")[0]
        split_lang["validation"] = lang.split("-")[1]
        split_lang["test"] = lang.split("-")[1]
    else:
        split_lang["train"] = lang
        split_lang["validation"] = lang
        split_lang["sample"] = lang
        split_lang["test"] = lang

    def load_data(split_names):
        myds = {}
        for split_name in split_names:
            df_path = split_path[split_name]
            split_df = pd.read_table(df_path)
            mlog.info("Path of dataset for %s %s", split_name, split_path[split_name])
            dlog.info("Columns of %s\n  %s", split_name, "\n".join(list(split_df.columns)))
            dlog.info(split_df.head())
            slang = split_lang[split_name]
            if "2" in slang:
                inp_lang, targ_lang = slang.split("2")
            else:
                inp_lang = targ_lang = slang
            inp_include, inp_exclude = filter_inputs(include, exclude, inp_lang)
            targ_include, targ_exclude = filter_inputs(include, exclude, targ_lang)
            mlog.info("Creating dataset for %s", split_name)
            # dddddddddddddd
            myds[split_name] = MyDataset(split_df, split_name,
                                method, prompt_pos, rel_filter,
                                num_samples[split_name], 
                                ignore_blanks,
                                only_blanks,
                                inp_include,
                                inp_exclude,
                                targ_include,
                                targ_exclude,
                                pred_tresh, nli_group, per_record, is_even, start, 
                                sampling, ex_type,
                                samples_per_head, save_ds_path[split_name]
                        )
        return myds

    if do_eval:
        myds = load_data([test_set])
        val_records = myds[test_set].num_records
        train_records = 0
    else:
        myds = load_data(["train", "validation", "sample"])
        samples_iter = iter(myds["sample"])
        _sample = True
        generate_samples["sample"] = []
        dlog.info("----------- SAMPLES -------------")
        while _sample:
            _sample = next(samples_iter, None)
            dlog.info(_sample)
            if _sample:
                generate_samples["sample"].append((_sample[0], _sample[1]))
        dlog.info("--------------------------------")
        mlog.info("Preparing samples: %s ", len(generate_samples["sample"]))
        train_records = myds["train"].num_records

    for logger in [mlog, clog, vlog]:
        logger.info("Train records:"  + str(train_records))
    if not do_eval:
        assert train_records != 0, "There is no data to train!!!!!!!!"
    if model_id == "test":
        return
    mlog.info("len tokenizer %s", len(tokenizer))
    my_specials = [x for x in tokenizer.additional_special_tokens if not "<extra_id"  in x]
    mlog.info("list of spcial tokens: %s", my_specials)
    extra = "_" + now
    m_name = model_id + "-" + method
    if do_eval:
        m_name = model_id + "-EVAL"
        exp_info = {"exp":experiment, "model":model_id, "lang": lang, 
                        "method":method, "wrap": w_str + "-" + encoder_type,
                        "frozen":f_str, 
                        "epochs":f"tr:{training_round}-ep:{epochs_num}-({start}-{train_records})-{val_records}", "date":extra}
    if do_eval or (not wrap and frozen):
        mlog.info("Evaluating the model...")
        model.to(device=device)
        evaluate(model, tokenizer, myds[test_set], underlying_model_name, exp_info, val_records, gen_param, no_score=no_score)  
        return
    accumulation_tiny_steps = 2 
    if "gpt" in model_id:
        accumulation_tiny_steps = 1
    if batch_size >= 2:
        node_batch_size = batch_size//accumulation_tiny_steps
    else:
        accumulation_tiny_steps = 1
        node_batch_size = 1

    iterations = train_records//batch_size
    
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
        queries,responses,rel,lang, index = zip(*batch)
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
         queries,responses,_,_,_ = zip(*batch)
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

    train_dataloader = torch.utils.data.DataLoader(myds['train'],
        batch_size=node_batch_size,shuffle=shuffle,collate_fn=data_collator, num_workers=num_workers)
    dev_dataloader = torch.utils.data.DataLoader(myds['validation'],
        batch_size=node_batch_size,shuffle=shuffle,collate_fn=data_collator)
    # %% prepare for training
    sw = SummaryWriter(save_path, flush_secs=1)
    no_decay = ['bias', 'LayerNorm.weight']
    if wrap and not frozen:
        raise "Are you sure you want to wrap without freezing the model?"
    wrapped_model = None
    if wrap:
        if not load_prompt_path and Path(os.path.join(load_path, model_id, "prompt")).exists():
            load_prompt_path = os.path.join(load_path, model_id, "prompt")
            mlog.info("prompt path:%s ", load_prompt_path)
        mlog.info("Wrapping the model ...")
        wrapped_model = wrap_model(model, tokenizer, encoder_type, load_prompt_path, from_words = from_words, merge_prompts=merge_prompts, method = method) 
    if wrapped_model:
        wrapped_model.to(device=device)
        wrapped_model.prompt_encoders.to(device=device)
        mlog.info("len tokenizer after wrapping %s", len(tokenizer))
    else:
        wrap = False
        if learning_rate > 0.0001:
            raise "Learning rate should be smaller"
        extend_tokenizer(tokenizer)
        mlog.info("len tokenizer after extending %s", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))
        model.to(device=device)

    if frozen:
        for p in model.parameters():
            p.requires_grad = False 

    if wrap:
        #model.get_input_embeddings().weight.requires_grad = False
        rgrad = [p for p in wrapped_model.parameters() if p.requires_grad]
        nrgrad = [p for p in wrapped_model.parameters() if not p.requires_grad]

        _sum = 0
        for encoder in wrapped_model.prompt_encoders:
            enc_rgrad = [p for p in encoder.parameters() if p.requires_grad]
            tlog.info("Encoder require grad %s: %s",encoder.name,enc_rgrad)
            tlog.info("len Encoder require grad %s: %s",encoder.name,len(enc_rgrad))
            tlog.info("Encoder prompt ids: %s", encoder.prompt_ids)
            _sum += len(encoder.prompt_ids)
            tlog.info("len prompt ids %s: %s",encoder.name, len(encoder.prompt_ids))

        tlog.info("_sum: %s", _sum)
        model_rgrad = [p for p in model.parameters() if p.requires_grad]
        model_nrgrad = [p for p in model.parameters() if not p.requires_grad]
        tlog.info("Model require grad %s, ", len(model_rgrad))
        tlog.info("Model param that requires grad %s", model_rgrad)
        tlog.info("Model not require grad %s, ", len(model_nrgrad))
        tlog.info("Wrapped model require grad %s, ", len(rgrad))
        tlog.info("Wrapped model not require grad %s, ", len(nrgrad))

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
    def consume(iterator, n):
        '''Advance the iterator n-steps ahead. If n is none, consume entirely.'''
        collections.deque(itertools.islice(iterator, n), maxlen=0)

    #%% tttttt
    mlog.info("batch size: %s", batch_size)
    mlog.info("node batch size: %s", node_batch_size)
    if epochs_num == 0 or not wrap and frozen:
        mlog.info("Skip training...")
    elif step <= iterations and (wrap or not frozen):
        mlog.info("Training... %s", save_path)
    pbar = tqdm(total=iterations, position=0, leave=True) #,dynamic_ncols=True)
    for epoch in range(epochs_num):
        train_iter = iter(train_dataloader)
        mlog.info("Saving train data set...")
        myds["train"].save()
        mlog.info(f"============== epoch {epoch}\n")
        tlog.info(f"============== epoch {epoch}\n")
        tot_loss = 0
        step = 0
        if train_start > 0:
            mlog.info("skipping %s", train_start)
            consume(train_iter, train_start)
            pbar.update(train_start)
            step = train_start
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
                        mlog.info('dev/micro_avg_loss: %s-%s',dev_micro_avg_loss,step)
                        sw.add_scalar('dev/macro_avg_loss',dev_macro_avg_loss,step)
                        vlog.info('dev/macro_avg_loss: %s-%s',dev_macro_avg_loss,step)
                        mlog.info('dev/macro_avg_loss: %s-%s',dev_macro_avg_loss,step)
                        if dev_micro_avg_loss < best_dev_loss:
                            best_dev_loss = dev_micro_avg_loss
                            best_eval_step = step
                            tlog.info("epoch %s, best_eval_step: %s", epoch, best_eval_step)
                            save_checkpoint(model, tokenizer, optimizer, scheduler, step, 
                                            best_eval_step, best_dev_loss,
                                            os.path.join(save_path, "best_model"))

                            generation_results = \
                            "|Queries|Generation Results|\n"\
                            "|-|-|\n"
                            for i,(_q,_target) in enumerate(generate_samples['sample']):
                                if i==validation_num_generation:
                                    break
                                results = generate(model, tokenizer, [_q.strip()]) 
                                vlog.info("%02d) %-50s | %-50s | %-40s", i, _q.strip(), 
                                        results, _target.strip())
                                generation_results+=f"|`{_q},{_target}`|`{str(results)}`|\n"
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
                if wrap:
                    tlog.info("Wrap model zero grad")
                    wrapped_model.zero_grad()
                else:
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
                    tlog.info("Original embedding grads:%s",model.get_input_embeddings().weight.grad)
                    if wrap:
                        #tlog.info("Merge embedding grads:%s", wrapped_model.merge_encoder.embedding.weight.grad)
                        for encoder in wrapped_model.prompt_encoders:
                            if encoder.name == "xIntent":
                                timelog.info("---------------- %s ---------------", encoder.name)
                                timelog.info("Prompt embedding grads:%s", encoder.embedding.weight.grad)

                    batch_loss += loss.item()
                optimizer.step()
                scheduler.step()
                step+=1
                bloss = batch_loss.item()
                tot_loss += bloss
                mean_loss = tot_loss/(step-train_start)
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
        with torch.no_grad():
            mlog.info("Updating the model weights before evaluaton...")
            wrapped_model.update_model_weight()
    model.eval()
    if not no_save_model:
        save_checkpoint(model, tokenizer, optimizer, scheduler, step, 
                        best_eval_step, best_dev_loss,
                        save_path)
    else:
        mlog.info("No save model is on!!")

    #% vvvvvvvvvvvvvvvv
    if test_set:
        myds = load_data([test_set])
        val_records = myds[test_set].num_records
        exp_info = {"exp":experiment, "model":model_id, "lang": lang, 
                        "method":method, "wrap": w_str + "-" + encoder_type,
                        "frozen":f_str, 
                        "epochs":f"tr:{training_round}-ep:{epochs_num}-({start}-{train_records})-{val_records}", "date":extra}
        evaluate(model, tokenizer, myds[test_set], save_path, exp_info, val_records, gen_param, attention_mask, no_score=no_score)  
    else:
        mlog.info("Test set was not provided.... skip testing...")
        

#ettt

@run.command()
@click.argument("experiment", type=str)
@click.option(
    "--model_ids",
    "-m",
    default="t5-base",
    type=str,
    help=""
)
@click.option(
    "--no_score",
    "-ns",
    is_flag=True,
    help=""
)
@click.option(
    "--keep",
    "-k",
    is_flag=True,
    help="keep old experiments"
)
@click.option(
    "--colab",
    "-c",
    is_flag=True,
    help="Force settings for colab"
)
def exp(experiment, model_ids, no_score, keep, colab):
    #cccccccccccc
    if colab:
        pretPath = "/content/drive/MyDrive/pret"
        logPath = "/content/"
        resPath = "/content/drive/MyDrive/pouramini/results"
    else:
        logPath = os.path.join(home, "logs")
        resPath = os.path.join(home, "results") 
        pretPath = os.path.join(home, "pret") 

    base_dir = home
    if "_" in experiment:
        raise ValueError("Experiment name shouldn't have underscore in it, use dash")
    conf = os.path.join(base_dir, "logs/confs/exp_conf.json")
    save_path = os.path.join(base_dir, "mt5-comet/comet/train/")
    if not colab:
        conf_path = os.path.join(save_path,"confs")
    else:
        conf_path = os.path.join(save_path,"colab_confs")
    mlog.info("Creating configurations...%s ", conf_path)
    if not keep:
        cur_files = glob.glob(f"{conf_path}/*")
        mlog.info("Cleaning previous exps ... %s", len(cur_files))
        for f in cur_files: 
            os.remove(f)
    Path(conf_path).mkdir(exist_ok=True, parents=True)
    if Path(conf).exists():
       with open(conf, 'r') as f:
           args = json.load(f) 
    else:
        print(conf + " doesn't exists!")
        return
    samples = 300
    args["experiment"] = experiment
    args["cycle"] = 0
    args["no_save_model"] = True if colab else False
    args["no_score"] = no_score
    args["load_path"] = pretPath
    args["train_path"] = "atomic/train.tsv"
    save_path = os.path.join(pretPath, experiment)
    args["save_path"] = save_path

    args["cpu"] = False 
    args["config"] = False 
    args["batch_size"] = 16 if colab else 4 
    args["gen_param"] = "greedy" 
    args["exclude"] = "natural" 
    langs = {"en":True}
    args["test_samples"] = 4500 
    methods = {"sup-tokens":"u","sup":"u", "sup-nat":"u","unsup":"u","unsup-tokens":"w-u","unsup-nat":"u", "sup-nat-tokens":"u","unsup-nat-tokens":"u"}
    samples_list = [270,2700, 27000]
    ii = 0
    models = model_ids.split("#")
    for model in models:
        for method,wrap in methods.items():
            for wrap in wrap.split("-"): 
                for samples in samples_list: 
                   w = "wrapped" if wrap == "w" else "unwrapped"
                   args["method"] = method
                   args["train_samples"] = samples
                   args["is_even"] = False
                   args["model_id"]= model
                   args["frozen"] = False
                   if w == "wrapped":
                       args["frozen"] = True
                   args["wrap"] = ""
                   if w == "wrapped":
                       args["wrap"] = True
                       args["batch_size"] = 20 if not colab else 48 
                   name = f"{experiment}-{model}-{samples}-{method}-{w}"
                   #name = name.replace("_unwrapped", "")
                   #name = name.replace("_unfreezed", "")
                   args["output_name"] = name
                   args["overwrite"] = name
                   ii +=1
                   name = "{:02d}".format(ii) + "_" + name
                   print(name)
                   with open(os.path.join(conf_path, f'{name}.json'), 'w') as outfile:
                            json.dump(args, outfile, indent=4)


def translate(model, tokenizer, df, trans_col, path, logger=None, start=0, save_path=""):
    pbar = tqdm(total= len(df))
    oldcol, newcol,save_step = trans_col.split("@")
    newcol = oldcol + "_" + newcol
    if newcol in df:
        mlog.info("column %s already exists... skipping", newcol)
        return df
    save_step = int(save_step)
    trans = []
    mlog.info("len(df): %s", len(df))
    mlog.info("save_path: %s", save_path)
    fname = Path(path).stem
    ii = 0
    first = True
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
            mlog.info("len(trans): %s", len(trans))
            mlog.info("row: %s", trans_row)
        pbar.update()
        if len(trans) > 3 and len(trans) % save_step == 0 or (len(trans) == 5 and first):
            p = os.path.join(save_path, fname + str(ii).replace("000","k_") + ".tsv")
            mlog.info("Saving at %s", p)
            mlog.info("Len trans: %s", len(trans))
            new_df = pd.DataFrame(data=trans) 
            df[newcol] = new_df[newcol]
            first = False
            df.to_csv(p, sep="\t", index=False)
        ii += 1

    new_df = pd.DataFrame(data=trans) 
    df[newcol] = new_df[newcol]
    p1 = os.path.join(save_path,fname + str(ii).replace("000","k_") + ".tsv")
    p3 = os.path.join(logPath, fname + str(ii).replace("000","k_") + ".tsv")
    df.to_csv(p1, sep="\t", index=False)
    mlog.info("Saved at %s",p1)
    df.to_csv(p3, sep="\t", index=False)
    mlog.info("Saved at %s",p3)

    return df




if __name__ == "__main__":
   run()
