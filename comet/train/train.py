#%% load libraries
from comet.train.common import *
import itertools, collections
import shutil
from comet.train.eval import *
from comet.utils.dataset import TokenizedDataset
from comet.utils.configue import Configure
import comet.utils.tool as ut
from comet.models.unified.prefixtuning import Model
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import TrainingArguments
from comet.train.model import *
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

#gggggggggggggg
class MyCollator(object):
    def __init__(self, tokenizer, model, ds_type="train", prefix=False, model_type="t5"):
        self.tokenizer = tokenizer
        self.model = model
        self.ds_type = ds_type 
        self.prefix = prefix
        self.model_type = model_type
    def __call__(self, batch):
        queries,inputs, responses,rel,index,rep = zip(*batch)
        tokenizer = self.tokenizer
        rels = list(rel)
        desc = ["Predict the {}:".format(rel_nat_maps[x]["desc"]) for x in rels] 
        new_batch = tokenizer(list(queries),return_tensors='pt',padding='longest', 
                                #truncation=True, max_length=50
                             )
        if self.prefix:
            tokenized_description = tokenizer(desc,return_tensors='pt',padding='longest')
            tokenized_knowledge = tokenizer(rels,return_tensors='pt',padding='longest')

            new_batch['description_input_ids'] = torch.LongTensor(tokenized_description.data["input_ids"])
            new_batch['description_attention_mask'] = torch.LongTensor(tokenized_description.data["attention_mask"])
            new_batch['knowledge_input_ids'] = torch.LongTensor(tokenized_knowledge.data["input_ids"])
            new_batch['knowledge_attention_mask'] = torch.LongTensor(tokenized_knowledge.data["attention_mask"])
        if self.ds_type == "train": # or self.prefix:
            with tokenizer.as_target_tokenizer():
                tokenized_outputs = tokenizer(list(responses),return_tensors='pt',
                        padding='longest', 
                        #truncation=True, max_length=50
                        )
                labels = tokenized_outputs['input_ids']
                labels[labels==tokenizer.pad_token_id] = -100
                new_batch['labels']=labels
                if not self.prefix:
                    new_batch['decoder_input_ids'] = self.model.prepare_decoder_input_ids_from_labels(
                        tokenized_outputs['input_ids']
                    )
                    new_batch['decoder_attention_mask'] = tokenized_outputs['attention_mask']
        return new_batch
        def tokenize(self, batch):
            queries,inputs, responses,rel,index,rep = batch
            #queries = list(queries)
            #responses =list(responses)

            tokenized_inputs = tokenizer(queries,
                                    padding="max_length",
                                    truncation=True,
                                    max_length = 200,
                                     )
            input_ids = tokenized_inputs["input_ids"]
            new_batch = {}
            input_ids = torch.LongTensor(input_ids)
            #input_ids = input_ids.to(device)
            #input_ids = input_ids.unsqueeze(0)
            new_batch['input_ids'] = input_ids
            attention_mask = tokenized_inputs['attention_mask']
            attention_mask = torch.LongTensor(attention_mask)
            #attention_mask = attention_mask.unsqueeze(0)
            new_batch['attention_mask'] = attention_mask
            with tokenizer.as_target_tokenizer():
                tokenized_outputs = tokenizer(responses,
                                    padding="max_length",
                                    truncation=True,
                                    max_length = 200,
                                     )
                labels = tokenized_outputs['input_ids']
                labels[labels==tokenizer.pad_token_id] = -100
                labels = torch.LongTensor(labels)
                new_batch['labels']=labels
                #decoder_input_ids = model.prepare_decoder_input_ids_from_labels(
                #    tokenized_outputs['input_ids']
                #)
                #decoder_input_ids = torch.LongTensor('decoder_input_ids')
                #new_batch['decoder_input_ids'] = decoder_input_ids 
                #decoder_attention_mask = tokenized_outputs['decoder_attention_mask']
                #decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
                #new_batch['decoder_attention_mask'] = decoder_attention_mask 
            return new_batch
        def gpt_collate(batch):
             queries,responses,_,_,_,_ = zip(*batch)
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
             new_batch['labels']=labels
             return new_batch #,references
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Run    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@click.group()
def cli():
    pass
@cli.command(context_settings=dict(
            ignore_unknown_options=True,
            allow_extra_args=True,))
@click.option(
    "--conf_path",
    "-cp",
    default="",
    type=str,
    help=""
)
@click.option(
    "--base_conf",
    "-bc",
    default="base",
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
    "--exclude_conf",
    "-ex",
    default="",
    type=str,
    help=""
)
@click.option(
    "--include_conf",
    "-in",
    default="",
    type=str,
    help=""
)
@click.option(
    "--overwrite_conf",
    "-ow",
    is_flag=True,
    help=""
)
@click.option(
    "--var",
    "-var",
    default="",
    type=str,
    help=""
)
@click.option(
    "--save_model",
    "-sm",
    is_flag=True,
    help=""
)
@click.option(
    "--addto",
    "-at",
    default="",
    type=str,
    help=""
)
@click.option(
    "--rem",
    "-rem",
    is_flag=True,
    help="Remove old folder"
)
@click.option(
    "--save_data",
    "-sd",
    is_flag=True,
    help=""
)
@click.option(
    "--load_data",
    "-ld",
    is_flag=True,
    help=""
)
@click.option(
    "--add_prefix",
    "-ap",
    is_flag=True,
    help=""
)
@click.option(
    "--only_var",
    "-ov",
    is_flag=True,
    help=""
)
@click.option(
    "--sep",
    "-sep",
    default="/",
    type=str,
    help=""
)
@click.pass_context
#rrrrrrrrrrr
def run(ctx, conf_path, base_conf, experiment, 
        exclude_conf, include_conf, overwrite_conf, var, 
        save_model, addto, rem, save_data, load_data, add_prefix, only_var, sep):
     if not conf_path:
        conf_path = "confs"
        if colab: conf_path = "colab_confs"
     if ctx.invoked_subcommand is None:
        mlog.info("Reading from conf %s", conf_path)
        _path = f"{conf_path}/{experiment}"
        if not Path(_path).exists():
           conf = f"base_confs/{base_conf}.json" # default conf
           mlog.info("NEW experiment! Reading from conf %s", conf)
           if Path(conf).exists():
               with open(conf, 'r') as f:
                   args = json.load(f) 
           else:
               mlog.info(f"%s doesn't exists ...", conf)
               return
           args["config"] = False
           args["output_name"] = "" 
           if add_prefix:
               args["pre_prefix"] = experiment
           if addto:
               spath = os.path.join(pretPath, addto)
           else:
               spath = os.path.join(pretPath, experiment)
           if Path(spath).exists() and rem:
               #if input("Are you sure you want to delete the experiment folder?") == "y":
               shutil.rmtree(spath)
           Path(spath).mkdir(exist_ok=True, parents=True)
           if load_data:
               args["data_path"] = spath
           args["save_path"] = spath
           args["load_path"] = pretPath 
           _extra = ""
           exclude_list = ["no_confirm", "follow_method", "method", "test_samples"]
           mlog.info("Extra args=%s", ctx.args)
           for _item in ctx.args:
                mlog.info("arg = %s", _item)
                _key,_val = _item.split("=")
                _val = _val.strip()
                _key=_key.strip("--")
                if not _key in exclude_list:
                    _ks = "".join([k[0] for k in _key.split("_")])
                    _extra += "_" + (_ks + "_" + _val if not str(_val)=="True" else _key)
                mlog.info("set %s = %s", _key, _val)

                if _val == "null": 
                    args[_key]= ""

                if not _val == "False":
                    if _val == "True":
                        args[_key]= True
                    else:
                        args[_key]= _val
                else:
                   if _key in args:
                       del args[_key]
           # oooooooooooooo
           if not var:
               output_name = base_conf + sep + args["method"] + sep + _extra
               args["overwrite"] = output_name
               args["no_save_model"] = not save_model
               ctx.invoke(train, **args)
           else:
               output_name = ""
               var = var.replace("all_rels","#".join(all_rels))
               var = var.replace("x_rels","#".join(x_rels))
               all_vars = var.split("--")
               #all_vars = sorted(all_vars)
               var_names = [x.split("=")[0] for x in all_vars]
               values = [x.split("=")[1].split("#") for x in all_vars]
               tot_comb = [dict(zip(var_names, comb)) for comb in itertools.product(*values)]
               ii = 0
               for comb in tot_comb:
                   _output_name = output_name
                   __output_name = output_name
                   for var_name,var_item in comb.items():
                       if var_item == "null": var_item = ""
                       if var_item.strip() == "False": 
                           if var_name in args:
                               del args[var_name]
                       else:
                           if var_item.strip() == "True":
                               var_item = True
                           if not var_name in exclude_list:
                               _output_name +=  sep + var_name + "_" + str(var_item)
                           __output_name +=  sep + var_name + "_" + str(var_item)
                           args[var_name]=var_item
                   ii += 1
                   rel_folder = "all" if not args["rel_filter"] else args["rel_filter"]
                   if only_var:
                       args["overwrite"] = __output_name.strip(sep)
                   else:
                       args["overwrite"] = args["method"] + sep + rel_folder + sep + _output_name \
                           + sep + _extra 
                   args["no_save_model"] = not save_model
                   if save_data: 
                       args["save_data"] = spath
                   if args["rel_filter"] == "multi":
                       args["data_path"] = spath
                       args["rel_filter"] = "" 
                       args["pre_prefix"] = experiment 
                   else:
                       args["data_path"] = ""
                       args["pre_prefix"] = "" 

                   ctx.invoke(train, **args)
        else:
            confs = sorted(glob.glob(f"{_path}/*"))
            default_model = ""
            first = True
            for conf in confs:
                fname = Path(conf).stem
                mlog.info(f"%%% {fname} %%%")
                if not experiment in fname:
                    mlog.info("Skipping .... This was not in experiments")
                    continue
                if exclude_conf and exclude_conf in fname:
                    mlog.info("Skipping .... by exclude")
                    continue
                if include_conf and not include_conf in fname:
                    mlog.info("Skipping .... by include")
                    continue
                if Path(conf).exists():
                   with open(conf, 'r') as f:
                       args = json.load(f) 
                   spath = args["save_path"]
                   if first and Path(spath).exists():
                       mlog.info("%s already exists!", spath)
                       first = False
                       if overwrite_conf:
                          shutil.rmtree(spath)

                   Path(spath).mkdir(exist_ok=True, parents=True)
                   #mlog.info("save path: %s", spath)
                   cur_res_path = os.path.join(spath, args["output_name"], "full_result*")
                   #mlog.info("cur_res_path: %s", cur_res_path)
                   cur_res = glob.glob(cur_res_path)
                   #mlog.info("cur_res: %s", cur_res)
                   if cur_res:
                        mlog.info("Skipping .... This was done before %s ", spath)
                        continue
                   for i in range(0, len(ctx.args), 2):
                        _key = ctx.args[i][2:]
                        if _key in args:
                            args[_key]= ctx.args[i+1] 
                   ctx.invoke(train, **args)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@cli.command()
@click.option(
    "--model_id",
    "-mid",
    default="t5-base",
    type=str,
    help=""
)
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
    "--sample_samples",
    "-sn",
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
    "--val_method",
    "-vmt",
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
    "--prefix",
    "-pt",
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
    "--data_path",
    "-dp",
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
    default="8",
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
    default=3,
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
    default="mlp",
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
    default="xIntent",
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
    "--scorers",
    "-nos",
    default="rouge-bert",
    type=str,
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
@click.option(
    "--gen_bs",
    "-gb",
    default="30@10",
    type=str,
    help="Batch sizes for generation."
)
@click.option(
    "--shared_embs",
    "-she",
    is_flag=True,
    help=""
)
@click.option(
    "--no_confirm",
    "-nc",
    is_flag=True,
    help="Don't ask confirmations"
)
@click.option(
    "--follow_method",
    "-fm",
    is_flag=True,
    help="Adjust some settings like wrapping or freezing the model according to the method"
)
@click.option(
    "--repeat",
    "-rep",
    default=1,
    type=int,
    help="How many a training example must be repeated"
)
@click.option(
    "--trial",
    "-try",
    default=1,
    type=int,
    help="Repeating the same experiment with different tirals"
)
@click.option(
    "--fz_parts",
    "-fzl",
    default="",
    type=str,
    help="Layers to be freezed"
)
@click.option(
    "--pid",
    "-pid",
    default=0,
    type=int,
    help="Prompt id or index for templates that have multiple prompt templates"
)
@click.option(
    "--use_dif_templates",
    "-udt",
    is_flag=True,
    help="Wether to use different templates (repeat must be greater or equal to the number of various templates)"
)
@click.option(
    "--break_sent",
    "-brk",
    is_flag=True,
    help="Break in input sentencet to input and target at different positions (based on how many times specified by repeat)"
)
@click.option(
    "--sort",
    "-sk",
    default="event",
    type=str,
    help="sort key"
)
@click.option(
    "--do_preproc",
    "-dop",
    is_flag=True,
    help=""
)
@click.option(
    "--replace_blanks",
    "-rb",
    is_flag=True,
    help=""
)
@click.option(
    "--loop",
    "-tl",
    is_flag=True,
    help=""
)
@click.option(
    "--know",
    "-ku",
    default="s",
    type=str,
    help=""
)
@click.option(
    "--show_samples",
    "-ss",
    is_flag=True,
    help=""
)
@click.option(
    "--ph_num",
    "-pn",
    default=3,
    type=int,
    help="Repeat of placeholder"
)
@click.option(
    "--save_data",
    "-sd",
    default="",
    type=str,
    help=""
)
@click.option(
    "--pre_prefix",
    "-pre",
    default="",
    type=str,
    help=""
)
@click.option(
    "--skip",
    "-skip",
    is_flag=True,
    help="Skip the experiment if it already exists."
)
def train(model_id, experiment, qtemp, anstemp, extemp, method, val_method, train_samples, test_set, val_samples, test_samples, sample_samples, load_path, data_path, train_path, val_path, test_path, sample_path, overwrite, save_path, output_name, lang, pred_tresh, ignore_blanks,only_blanks, include, exclude, nli_group, learning_rate, do_eval, cont, wrap, prefix, frozen, freez_step, unfreez_step, cpu, load_prompt_path, verbose, cycle, batch_size, path, from_dir, is_flax, config,clear_logs, gen_param, print_log, training_round, epochs_num, per_record, is_even, start, prompt_length, prompt_pos, zero_shot, sampling, opt_type, samples_per_head, deep_log, trans, encoder_type, from_words,rel_filter, ex_type, last_data, save_df, merge_prompts, num_workers, scorers, train_start, no_save_model, gen_bs, shared_embs, no_confirm, follow_method, repeat, trial, fz_parts, pid, use_dif_templates, break_sent,sort, do_preproc, replace_blanks, loop, know, show_samples, ph_num, save_data, pre_prefix, skip):

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

    mlog.info("========================= Version 8 ========================")
    if save_path == "":
        if "ahmad" or "pouramini" in home:
            save_path = os.path.join(home, "logs")
        else:
            save_path = "/content/drive/MyDrive/pouramini/logs"

    if fz_parts == "none":
        frozen = False
        fz_parts = ""

    if "-wrap" in method and not wrap:
        mlog.info("Method %s is for wrapped models...", method)
        wrap = True
        if wrap and not frozen and follow_method:
            frozen = True
    if wrap and not frozen:
         if not no_confirm:
             ans = input("Are you sure you want to wrap without freezing the model?")
             if ans != "y":
                 frozen = True
    w_str = "wrapped" if wrap else "unwrapped"
    f_str = "freezed" if frozen else "unfreezed"
    if not output_name and not (cont or do_eval):
        output_name = model_id + "_" + method 
    conf_path = save_path
    if model_id == "test":
        save_path = ""
        output_name = "test"
    if config:
        conf_path = "base_confs"
        Path(conf_path).mkdir(exist_ok=True, parents=True)
        args["config"] = False
        args["output_name"] = ""
        with open(os.path.join(conf_path, f'{output_name}.json'), 'w') as outfile:
            json.dump(args, outfile, indent=4)

        mlog.info("Config %s was created at %s", output_name + ".json", conf_path)
        return

    if data_path:
        train_path = os.path.join(data_path, "train.tsv")
        test_path = os.path.join(data_path, "test.tsv")
        val_path = os.path.join(data_path, "validation.tsv")
        sample_path = os.path.join(data_path, "sample.tsv")
        train_samples, test_samples = 0, 0

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
    learning_rate = float(learning_rate)
    lr_mt = {}
    if not frozen and learning_rate == 0: 
        if opt_type == "adam": 
            learning_rate = lr_mt[method] if method in lr_mt else 6.25e-05  
        else:
            learning_rate = 1e-3
        if "gpt" in model_id:
            learning_rate = 1e-5
    if frozen and learning_rate == 0: 
        if encoder_type == "lstm":
            learning_rate = 0.05  
        elif encoder_type == "emb":
            learning_rate = 0.1  
        else:
            learning_rate = 0.01  

    assert learning_rate > 0, "Learning rate is zero!"
    device = 'cuda' if not cpu else 'cpu'
    mlog.info("Optimizer type %s:", opt_type)
    mlog.info("learning rate %s:", learning_rate)

    log_dir = save_path
    set_device(device)
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
            conf_path = os.path.join(save_path, "exp_conf.json")
            if Path(conf_path).is_file():
               with open(conf_path, 'r') as f:
                   args = json.load(f) 
               mlog.info(args)
               mlog.info("Loading from configuration file")
               qtemp = args['qtemp']
               atemp = args['anstemp']
               mlog.info("Qtemp: %s", args['qtemp'])
               mlog.info("Anstemp: %s", args['anstemp'])

    args_str = json.dumps(args, indent=4)
    do_overwrite = False
    if overwrite:
        save_path = os.path.join(log_dir, overwrite)
        do_overwrite = True
    if Path(save_path).exists() and skip:
        mlog.info("Skiping.... the folder already exists!!")
        return
    ii = 1
    while not do_overwrite and Path(save_path).exists() and not model_id=="test":
        if not no_confirm and not do_eval:
            ans = input(f"The output directory {save_path} already exists, do you want to overwrite it? (y/n)")
        else:
            ans = "y"
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

    #save log files
    for logger, fname in zip([mlog,dlog,clog,vlog,tlog], ["main","data","cfg","eval","train"]):
        if len(logger.handlers) >= 3:
            continue
        logger.setLevel(logging.INFO)
        logFilename = os.path.join(save_path, fname + "_.log")
        handler = logging.FileHandler(logFilename, mode = "w" if clear_logs else "a")
        logger.addHandler(handler)

    for logger in [mlog, clog, dlog, tlog, vlog]:
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        logger.info(f"%%%%%%%%%%%%%%%%%% { model_id } ")
        logger.info(f"%%%%%%%%%%%%%%%%%% { now } ")
        logger.info(f"%%%%%%%%%%%%%%%%%% { output_name } ")
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    for logger in [clog]:
        logger.info(args_str)
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


    # save configurations
    with open(os.path.join(save_path, f'exp_conf.json'), 'w') as outfile:
        json.dump(args, outfile, indent=4)
    #%% load model

    def load_model(model_id, underlying_model_name):
        mlog.info("Loading model ...")
        if model_id == "test":
            return None, None, "test"
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
            tpath = underlying_model_name 
            if from_dir:
                tpath = f"{load_path}/{model_id}"
            tokenizer = AutoTokenizer.from_pretrained(tpath)
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
        length = [int(s) for s in str(prompt_length).split("-")]
        set_prompt_lengths(rel_filter, length)

    num_samples = {"train": int(train_samples), "validation":int(val_samples), "sample":int(sample_samples), "test":int(test_samples)}
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

    if not val_method:
        val_method = method
    def load_data(split_names):
        myds = {}
        for _name in split_names:
            name_opts = _name.split("+")
            split_name = name_opts[0]
            _replace_blanks = replace_blanks
            if len(name_opts) > 1:
                _replace_blanks = True
                _opt = name_opts[1]
                if _opt == "replace_blanks":
                    _replace_blanks = True

            tails_per_head = int(samples_per_head)
            #if "test" in split_name:
            #    tails_per_head = 0
            #    _replace_blanks = False
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
            _method = method
            _only_blanks = only_blanks
            if "test" in split_name:
                _method = val_method
                #_only_blanks = True
            _repeat = 1
            if split_name == "train" or split_name == "sample":
                _repeat = int(repeat)
            # dddddddddddddd
            myds[_name] = MyDataset(split_df, split_name,
                                _method, prompt_pos, rel_filter,
                                num_samples[split_name], 
                                ignore_blanks,
                                _only_blanks,
                                inp_include,
                                inp_exclude,
                                targ_include,
                                targ_exclude,
                                pred_tresh, nli_group, per_record, is_even, start, 
                                sampling, ex_type,
                                tails_per_head, save_ds_path[split_name], _repeat, 
                                int(pid), break_sent, sort, _replace_blanks, 
                                None, int(ph_num),
                        )
            if save_data:
                myds[_name].save_data(os.path.join(save_data,_name + ".tsv"), merge=True)
        return myds

    if do_eval:
        myds = load_data([test_set])
        val_records = myds[test_set].num_records
        train_records = 0
    else:
        if show_samples:
            ds_list = ["sample"]
        else:
            ds_list = ["train", "validation"]
            ds_list += ["sample"]
        myds = load_data(ds_list)
        if "sample" in ds_list:
            samples_iter = iter(myds["sample"])
            _sample = True
            generate_samples["sample"] = []
            if not show_samples:
                logger = dlog
            else:
                logger = mlog
            logger.info("----------- SAMPLES -------------")
            while _sample:
                _sample = next(samples_iter, None)
                logger.info(_sample)
                if False: #_sample:
                    generate_samples["sample"].append((_sample[0], _sample[1]))
            logger.info("--------------------------------")
            logger.info("Preparing samples: %s ", len(generate_samples["sample"]))
    if model_id == "test" or show_samples:
        return
    if not fz_parts or fz_parts == "all":
        modules_to_freeze = [model]
    else:
        modules_to_freeze = []

    _parts = fz_parts.split("@")
    if not "@" in fz_parts:
        enc_parts = _parts[0]
        freeze_self_att(modules_to_freeze, enc_parts, model.encoder)
    else:
        enc_parts = _parts[0]
        dec_parts = _parts[1]
        freeze_self_att(modules_to_freeze, enc_parts, model.encoder)
        freeze_self_att(modules_to_freeze, dec_parts, model.decoder, True)
    if len(_parts) == 3:
        decx_parts = _parts[2]
        freeze_cross_att(modules_to_freeze, decx_parts, model.decoder)

    def freeze(modules_to_freeze, fz=False):
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = fz  # Actual freezing operation
    if frozen:
        freeze(modules_to_freeze)

    mlog.info("len tokenizer %s", len(tokenizer))
    my_specials = [x for x in tokenizer.additional_special_tokens if not "<extra_id"  in x]
    vlog.info("list of spcial tokens: %s", my_specials)
    extra = "_" + now
    m_name = model_id + "-" + method
    p_str = "prefixed" if prefix else "not_prefixed"
    exp_info = {"exp":experiment, "model":model_id, "lang": lang, 
                    "method":method, 
                    "wrap": w_str + ("-" + encoder_type if wrap else ""),
                    "frozen":f_str, 
                    "prefixed":p_str,
                    "pre_prefix":pre_prefix,
                    "steps":train_samples,
                    "epochs":epochs_num,
                    "trial":trial,
                    "date":extra}
    exp_info["eval"] = do_eval
    if do_eval or (not wrap and frozen and modules_to_freeze is model):
        mlog.info("Evaluating the model...")
        model.to(device=device)
        evaluate(myds[test_set], underlying_model_name, exp_info, val_records, gen_param, scorers=scorers, batch_size=gen_bs, model=model, tokenizer=tokenizer)  
        return
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

    accumulation_tiny_steps = 2 
    batch_size = int(batch_size)
    if "gpt" in model_id:
        accumulation_tiny_steps = 1
    if batch_size >= 2:
        node_batch_size = batch_size//accumulation_tiny_steps
    else:
        accumulation_tiny_steps = 1
        node_batch_size = 1



# ggggggggg
    #%% build dataloader
    if "gpt" in model_id: 
        tokenizer.add_special_tokens(pad_token)
        tokenizer.add_special_tokens(sep_token)
        mlog.info("pad token id: %s", tokenizer.pad_token_id)
        data_collator = MyCollator(tokenizer, model, ds_type="train", model_type="gpt")
    else:
        data_collator = MyCollator(tokenizer, model, ds_type="train", prefix=prefix)

    train_dataset = myds["train"]#.map(tokenize)
    dev_dataset = myds["validation"]#.map(tokenize)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=node_batch_size,shuffle=shuffle, num_workers=num_workers,
        collate_fn=data_collator,
    )
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset,
        batch_size=node_batch_size,shuffle=shuffle,
        collate_fn=data_collator,
    )
    #torch.utils.data.DataLoader(myds['validation'],
    #    batch_size=node_batch_size,shuffle=shuffle,collate_fn=data_collator)
    train_records = myds["train"].num_records
    assert train_records != 0, "There is no data to train!!!!!!!!"
    for logger in [mlog, clog, vlog]:
        logger.info("Train records: %s", train_records)
    iterations = train_records//batch_size
    
    for logger in [mlog, tlog]:
        logger.info("Iterations:"  + str(iterations))
    warm_up_steps = 0.002*iterations
    #ppppppppppppppp
    if prefix:
        pre_args = Configure.Get("pl5.cfg")
        #mlog.info("prefix conf: %s", pre_args)

        pre_args.prefix_tuning.prefix_sequence_length = int(prompt_length)
        pre_args.model.knowledge_usage = 'separate' if know=="s" else "concatenate"
        model = Model(tokenizer, model, args=pre_args)
        model_tokenizer = model.tokenizer
        #model = ut.get_model(pre_args.model.name)(pre_args)
    # %% prepare for training
    sw = SummaryWriter(save_path, flush_secs=1)
    no_decay = ['bias', 'LayerNorm.weight']
    wrapped_model = None
    if wrap:
        if not load_prompt_path and Path(os.path.join(load_path, model_id, "prompt")).exists():
            load_prompt_path = os.path.join(load_path, model_id, "prompt")
            mlog.info("prompt path:%s ", load_prompt_path)
        mlog.info("Wrapping the model ...")
        model_to_wrap = model
        if prefix:
            model_to_wrap = model.pretrain_model
        wrapped_model = wrap_model(model_to_wrap, tokenizer, encoder_type, load_prompt_path, from_words = from_words, merge_prompts=merge_prompts, method = method, shared_embs= shared_embs) 
    if wrapped_model:
        wrapped_model.to(device=device)
        wrapped_model.prompt_encoders.to(device=device)
        mlog.info("len tokenizer after wrapping %s", len(tokenizer))
    else:
        wrap = False
        if learning_rate > 0.0001:
            #raise "Learning rate should be smaller"
            pass
        extend_tokenizer(tokenizer)
        mlog.info("len tokenizer after extending %s", len(tokenizer))
        if not prefix:
            model.resize_token_embeddings(len(tokenizer))
        else:
            model.pretrain_model.resize_token_embeddings(len(tokenizer))
        model.to(device=device)


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
        tlog.info("Wrapped model require grad %s, ", len(rgrad))
        tlog.info("Wrapped model not require grad %s, ", len(nrgrad))

    model_rgrad = [p for p in model.parameters() if p.requires_grad]
    model_nrgrad = [p for p in model.parameters() if not p.requires_grad]
    mlog.info("Model require grad %s, ", len(model_rgrad))
    mlog.info("Model not require grad %s, ", len(model_nrgrad))
    if not no_save_model:
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
            mlog.info("Ada Factor")
            # replace AdamW with Adafactor
#AdafactorOptimizer.beta1 = 0.0
#AdafactorOptimizer.clipping_threshold = 1.0
#AdafactorOptimizer.decay_rate = None
#AdafactorOptimizer.epsilon1 = 1e-30
#AdafactorOptimizer.epsilon2 = 0.001
#AdafactorOptimizer.factored = True
#AdafactorOptimizer.min_dim_size_to_factor = 128
#AdafactorOptimizer.multiply_by_parameter_scale = True
            optimizer = Adafactor(
                model.parameters(),
                lr=1e-3,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=0.0,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
		    ) 
        #optimizer = Adafactor(
        #        model.parameters(),
        #        lr=1e-3,
        #        eps=(1e-30, 1e-3),
        #        clip_threshold=1.0,
        #        decay_rate=-0.8,
        #        beta1=0.0,
        #        weight_decay=0.0,
        #        relative_step=False,
        #        scale_parameter=True,
        #        warmup_init=False
        #    )
            scheduler = AdafactorSchedule(optimizer)
        else:
            raise ValueError(opt_type + " must be one of adam, ada, ada_no_lr")
        return optimizer, scheduler

    optimizer, scheduler = get_optimizer(model, learning_rate, wrap, opt_type) 
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
    mlog.info(f"============== learning_rate {learning_rate}\n")
    mlog.info(f"============== frozen {frozen} {fz_parts} \n")
    mlog.info(f"============== prefixed {prefix}  \n")
    mlog.info(f"============== wrap {wrap}\n")
    epochs_num = int(epochs_num)
    def train_loop(epochs_num):
        train_iter = iter(train_dataloader)
        step = 0
        best_dev_loss = 100
        best_eval_step = 0
        if epochs_num == 0 or (not wrap and frozen and modules_to_freeze is model):
            mlog.info("Skip training...")
        elif step <= iterations and (wrap or not frozen or modules_to_freeze is not model):
            mlog.info("Training... %s", save_path)
        for epoch in range(epochs_num):
            tlog.info(f"============== epoch {epoch}\n")
            pbar = tqdm(total = iterations, position=0, leave=True) #,dynamic_ncols=True)
            if epoch > 0:
                train_iter = iter(train_dataloader)
            #mlog.info("Saving train data set...")
            #myds["train"].save()
            tot_loss = 0
            step = 0
            if train_start > 0:
                mlog.info("skipping %s", train_start)
                consume(train_iter, train_start)
                pbar.update(train_start)
                step = train_start
            while step < iterations-1:
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
                    #if unfreez_step > 0 and step > unfreez_step and froze:
                    #    mlog.info("unfreezing the model")
                    #    unfreez_step = 0
                    #    freeze(modules_to_freeze, True)
                    #    last_lr = scheduler.get_last_lr()[0]
                    #    optimizer, scheduler = get_optimizer(model, last_lr, wrap, opt_type)
                    #if freez_step > 0 and step > freez_step and not frozen:
                    #    mlog.info("freezing the model")
                    #    freez_step = 0
                    #    freeze(modules_to_freeze)
                    #    last_lr = scheduler.get_last_lr()[0]
                    #    optimizer, scheduler = get_optimizer(model, last_lr, wrap, opt_type)
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
                        #tlog.info("Original embedding grads:%s",model.get_input_embeddings().weight.grad)
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
    #% vvvv
    if loop: #not prefix:
       train_loop(epochs_num)
    else:
        training_args = TrainingArguments(output_dir=save_path)
        training_args.per_device_train_batch_size=node_batch_size
        training_args.num_train_epochs=epochs_num
        training_args.save_strategy="steps"
        training_args.save_stepsi=10000 
        training_args.save_total_limiti=1 

        #training_args.logging_steps=5
        training_args.learning_rate=learning_rate
        training_args.do_predict=True
        training_args.gradient_accumulation_steps=accumulation_tiny_steps
        train_dataset = myds["train"]#.map(tokenize)
        test_dataset = myds["sample"]
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            #optimizers = (optimizer, scheduler),
            #schedulers = schedulers,
            data_collator=data_collator,
            #evaluator=evaluator,
            # We name it "evaluator" while the hugging face call it "Metric",
            # they are all f(predictions: List, references: List of dict) = eval_result: dict
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            #test_dataset=test_dataset,
        )
        train_result = trainer.train()
    #vvv
    if test_set:
        if "@" in gen_bs:
            test_bs,_ = gen_bs.split("@")
        else:
            test_bs = int(gen_bs)

        test_bs = int(test_bs)
        for _set in test_set.split("@"):
            myds = load_data([_set])
            test_dataset = myds[_set]
            data_collator = MyCollator(tokenizer, model, ds_type="test", prefix=prefix)
            test_dataloader = torch.utils.data.DataLoader(test_dataset,
                batch_size=test_bs,shuffle=False, 
                collate_fn=data_collator,
            )
            mlog.info("Evaluating ... %s", _set)
            val_records = myds[_set].num_records
            exp_info["test_set"] = _set
            evaluate(test_dataset, test_dataloader, save_path, exp_info, val_records, gen_param, scorers = scorers, batch_size=gen_bs, model=model, tokenizer=tokenizer, set_name=_set)  
    else:
        mlog.info("Test set was not provided.... skip testing...")
        

#ettt

@cli.command()
@click.argument("experiment", type=str)
@click.option(
    "--model_ids",
    "-m",
    default="t5-base",
    type=str,
    help=""
)
@click.option(
    "--keep",
    "-k",
    is_flag=True,
    help="Keep old experiment files"
)
@click.option(
    "--server",
    "-s",
    default="Specify the server to run the experiment (e.g. colab)",
    type=str,
)
@click.option(
    "--exclude",
    "-ex",
    default="",
    type=str,
    help=""
)
@click.option(
    "--include",
    "-in",
    default="",
    type=str,
    help=""
)
@click.option(
    "--save_model",
    "-sm",
    is_flag=True,
    help=""
)
def exp(experiment, model_ids, keep, server, exclude, include, save_model):
    #cccccccccccc
    is_colab = colab or server == "colab"
    if is_colab:
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
    conf = "base_confs/test.json" 
    save_path = os.path.join(base_dir, "mt5-comet/comet/train/")
    if not is_colab:
        conf_path = os.path.join(save_path,"confs",experiment)
    else:
        conf_path = os.path.join(save_path,"colab_confs",experiment)
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
    var_list = []
    args["experiment"] = experiment
    args["cycle"] = 0
    args["no_save_model"] = not save_model
    args["load_path"] = pretPath
    args["train_path"] = "atomic/train.tsv"
    save_path = os.path.join(pretPath, experiment)
    args["save_path"] = save_path

    args["cpu"] = False 
    args["config"] = False 
    args["gen_param"] = "top_p" 
    langs = {"en":True}
    args["test_samples"] = 1000 
    #args["test_path"] = "atomic/val_all_rels.tsv"
    methods = {"sup-tokens":"u","sup":"u", "sup-nat":"u","unsup":"u","unsup-tokens":"u","unsup-nat":"u", "sup-nat-tokens":"u","unsup-nat-tokens":"u", "sup-wrap":"w", "unsup-wrap":"w", "unsup-wrap-nat":"w", "unsup-tokens-wrap":"w", "sup-tokens-wrap":"w"}
    methods = {"sup-tokens-wrap":"w", "unsup-tokens-wrap":"w"} #, "sup-tokens-wrap":"w", "unsup-tokens-wrap":"w"}
    only_wrapped = True
    args["encoder_type"] = "mlp@2@200"
    extra = "mtype"
    var_list = ["mlp","mlp@2@200", "lstm"] #, "mlp@2@200", "lstm@2@200", "mlp@1@200", "mlp@1@1000"] #, 36000] #samples
    var_name = "encoder_type"
    args["train_samples"] = 2700
    ii = 0
    models = model_ids.split("#")

    def fill_args(model, method, wu, ii, var_name="", var=""):
       w = "wrapped" if wu == "w" else "unwrapped"
       if var_name:
           args[var_name] = var
       args["method"] = method
       args["is_even"] = False
       args["model_id"]= model
       args["frozen"] = False
       if w == "wrapped":
           args["frozen"] = True
       args["wrap"] = w == "wrapped" 
       args["batch_size"] = 16 if is_colab else 8 
       if w == "wrapped":
           args["wrap"] = True
           if model == "t5-base":
               args["batch_size"] = 20 if not is_colab else 40 
               args["gen_bs"] = "30@10" if not is_colab else "30@15" 
           else:  
               args["batch_size"] = 10 if not is_colab else 40 
               args["gen_bs"] = "5@1" if not is_colab else "30@10" 
       else:
           if only_wrapped:
               return ii
           if model == "t5-large":
               return ii
       if var_name:
           name = f"{experiment}-{model}-{method}-{var_name}-{var}"
       else:
           name = f"{experiment}-{model}-{method}"
       if extra:
           name += f"@{extra}"
       if include and not include in name:
           #mlog.info("Skipping by include ... %s", include)
           return ii
       if exclude and exclude in name:
           #mlog.info("Skipping by include ... %s", exclude)
           return ii
       #name = name.replace("_unwrapped", "")
       #name = name.replace("_unfreezed", "")
       name = "{:02d}".format(ii) + "_" + name
       args["output_name"] = name
       args["overwrite"] = name
       print(name)
       with open(os.path.join(conf_path, f'{name}.json'), 'w') as outfile:
                json.dump(args, outfile, indent=4)
       return ii + 1

    ii = 0
    for model in models:
        for method,wrap in methods.items():
            for wu in wrap.split("-"): 
                if var_list:
                    for var in var_list: 
                        ii = fill_args(model, method, wu,ii, var_name, var)
                else:
                    ii = fill_args(model, method, wu, ii)

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
   cli()
