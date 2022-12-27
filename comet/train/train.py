#%% load libraries
import wandb
import debugpy
import comet.train.mylogs as mylogs 
from comet.train.common import *
from comet.train.dataset import *
from comet.train.data import *
#from comet.data import TaskDataCollatorForSeq2Seq
#from comet.data import AutoTask
from comet.utils.utils import (modify_model_after_init, 
        save_training_config, save_prompts,get_adapter_config)

from comet.metrics.metrics import TASK_TO_METRICS
from comet.metrics.metrics import build_compute_metrics_fn
import comet.metrics.metrics as metrics
import itertools, collections
import shutil
from comet.train.eval import *
import comet.third_party.models as tp
from comet.utils.dataset import TokenizedDataset
from comet.utils.configue import Configure
import comet.utils.tool as ut
#from comet.models.unified.prefixtuning import Model
from comet.third_party.trainers import Seq2SeqTrainer
from transformers.optimization import Adafactor
from torch.optim import SparseAdam
from transformers import TrainingArguments, HfArgumentParser
from comet.options import AdapterTrainingArguments, ModelArguments, DataTrainingArguments, TrainingArguments
from comet.train.model import *
from comet.data_utils import *
from comet.polytropon import SkilledMixin
import torch
import re
import json
import glob
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler
from comet.samplers import RandomSampler
import os,time
import argparse
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import click
from tqdm import tqdm


class MyCollator(object):
    def __init__(self, tokenizer, model, ds_type="train", prefix=False, model_type="t5"):
        self.tokenizer = tokenizer
        self.model = model
        self.ds_type = ds_type 
        self.prefix = prefix
        self.model_type = model_type
        self.prompt_config = None

#gggggggggggggg
    def collate(self, enc_qs, enc_resps, queries, resps, targets, flags, tasks):
        bs = len(enc_qs)
        q_len = [len(i) for i in enc_qs]
        max_enc_len = max(q_len)
        r_len = [len(i) for i in enc_resps]
        max_dec_len = max(r_len) + 2 
        pad_id = self.tokenizer.pad_token_id
        all_rels = list(rel_maps.keys())
        model_data = {
            "input_ids": torch.ones(bs, max_enc_len, dtype=torch.long) * pad_id,
            "attention_mask": torch.zeros(bs, max_enc_len),
            "decoder_attention_mask": torch.zeros(bs, max_dec_len),
            #"cross_attention_mask": torch.zeros(bs, 1, max_dec_len, max_enc_len),
            "decoder_input_ids": torch.ones(bs, max_dec_len, dtype=torch.long) * pad_id,
            "labels": torch.ones(bs, max_dec_len, dtype=torch.long) * -100,
            "task":torch.ones(bs)*-1,
        }
        no_model_data = {
            #"idx": torch.zeros(bs, dtype=torch.long),
            "labels": torch.ones(bs, max_dec_len, dtype=torch.long) * pad_id,
            "loss_mask": torch.zeros(bs, max_dec_len),
            "query":[""]*bs,
            "target":[""]*bs,
            "task":torch.ones(bs)*-1,
            "resp":[-1]*bs,
            "wrap":[False]*bs,
            "freeze":[False]*bs,
            "unfreeze":[False]*bs,
            "method":[""]*bs
        }
        for i, (q, r, query, resp, target, flag, task) in enumerate(zip(enc_qs, enc_resps, queries, resps, targets, flags, tasks)):
            dec_ids = [pad_id] + r #+ [self.tokenizer.eos_token_id]
            label = r[:-1] #+ [self.tokenizer.eos_token_id]
            model_data["input_ids"][i][:len(q)] = torch.tensor(q, dtype=torch.long)
            model_data["decoder_input_ids"][i][:len(dec_ids)] = torch.tensor(dec_ids, dtype=torch.long)
            model_data["attention_mask"][i][:len(q)] = 1.0
            model_data["decoder_attention_mask"][i][:len(dec_ids)] = 1.0 
            model_data["task"][i] = task
            no_model_data["task"][i] = task
            #model_data["cross_attention_mask"][i][0, :dec_len, :enc_len] = 1.0 
            #no_model_data["idx"][i] = samp["idx"]
            model_data["labels"][i][:len(label)] = torch.tensor(label, dtype=torch.long)
            no_model_data["labels"][i][:len(label)] = torch.tensor(label, dtype=torch.long)
            no_model_data["query"][i] = query 
            no_model_data["target"][i] = target 
            no_model_data["wrap"][i] = flag["wrap"] 
            no_model_data["freeze"][i] = flag["freeze"] 
            no_model_data["unfreeze"][i] = flag["unfreeze"] 
            no_model_data["method"][i] = flag["method"] 
            no_model_data["resp"][i] = resp
            if self.prompt_config is not None:
                no_model_data["loss_mask"][i][self.prompt_config["dec"]["prompt_len"]:len(label)] = 1.0
            else:
                no_model_data["loss_mask"][i][:len(label)] = 1.0

        loop =  logs.args("loop")
        if loop == "custom":
            return model_data , no_model_data
        else:
            return model_data

    def __call__(self, batch):
        #return {"query":_query, "event":event, "resp":response, "rel":rel, "index":index, "rep":rep}
        queries = []
        inputs = []
        responses = []
        tasks = []
        index = []
        rep = []
        dec_starts = []
        enc_queries = []
        enc_responses = []
        targets = []
        flags = []
        for b in batch:
            enc_query = self.tokenizer.encode(b["query"])
            queries.append(b["query"])
            enc_queries.append(enc_query)
            enc_resp = self.tokenizer.encode(b["resp"].strip())
            responses.append(b["resp"].strip())
            enc_responses.append(enc_resp)
            tasks.append(b["task"])
            inputs.append(b["event"].strip())
            targets.append(b["target"].strip())
            index.append(b["index"])
            rep.append(b["rep"])
            flags.append(b["flag"])

        return self.collate(enc_queries, enc_responses, queries, responses, targets, flags, tasks)
        #queries,inputs, responses,rel,index,rep = zip(*batch)
        no_model_batch = {}
        tokenizer = self.tokenizer
        new_batch = tokenizer(list(queries),return_tensors='pt',padding='longest', 
                                #truncation=True, max_length=50
                             )
        if self.prefix:
            rels = list(rel)
            desc = ["Predict the {}:".format(rel_nat_maps[x]["desc"]) for x in rels] 
            tokenized_description = tokenizer(desc,return_tensors='pt',padding='longest')
            tokenized_knowledge = tokenizer(rels,return_tensors='pt',padding='longest')

            new_batch['description_input_ids'] = torch.LongTensor(tokenized_description.data["input_ids"])
            new_batch['description_attention_mask'] = torch.LongTensor(tokenized_description.data["attention_mask"])
            new_batch['knowledge_input_ids'] = torch.LongTensor(tokenized_knowledge.data["input_ids"])
            new_batch['knowledge_attention_mask'] = torch.LongTensor(tokenized_knowledge.data["attention_mask"])
        if True: #self.ds_type == "train": # or self.prefix:
            with tokenizer.as_target_tokenizer():
                tokenized_outputs = tokenizer(list(responses),return_tensors='pt',
                        padding='longest', 
                        #truncation=True, max_length=50
                        )
                labels = tokenized_outputs['input_ids']
                #no_model_batch['labels']= labels.clone()
                loss_mask = labels.clone()
                loss_mask[loss_mask!=tokenizer.pad_token_id] = 1
                loss_mask[loss_mask==tokenizer.pad_token_id] = 0
                no_model_batch['loss_mask'] = loss_mask
                labels[labels==tokenizer.pad_token_id] = -100
                new_batch['labels']=labels#[:,:-1]
                no_model_batch['labels']=labels#[:,:-1]
                if not self.prefix:
                    pid = self.model.prepare_decoder_input_ids_from_labels(
                        tokenized_outputs['input_ids'] 
                    )
                    new_batch['decoder_input_ids'] = pid
                    new_batch['decoder_attention_mask'] = tokenized_outputs['attention_mask']
        return new_batch, no_model_batch

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
    "--dpy",
    "-dpy",
    is_flag=True,
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
@click.option(
    "--num_exps",
    "-ne",
    default=0,
    type=int,
    help="number of experiments to be done, 0 means all"
)
@click.option(
    "--one",
    "-one",
    is_flag=True,
    help="only first experiment"
)
@click.option(
    "--cpu",
    "-cpu",
    is_flag=True,
    help="only use cpu"
)
@click.option(
    "--undone",
    "-ud",
    is_flag=True,
    help="List undone experiments"
)
@click.option(
    "--repeat",
    "-rep",
    is_flag=True,
    help="Repeat done expriments (not skip them)"
)
@click.option(
    "--port",
    "-p",
    default="1234",
    type=str,
    help="port for debugpy"
)
@click.option(
    "--wrap",
    "-wrap",
    is_flag=True,
    help=""
)
@click.option(
    "--break_point",
    "-bp",
    default="2",
    type=str,
    help="Stop on breakpoints equal to the value"
)
@click.option(
    "--reval_bests",
    "-best",
    is_flag=True,
    help=""
)
@click.option(
    "--trial",
    "-t",
    default="1",
    type=str,
    help="You can set it for repeating experiments with different identities"
)
@click.option(
    "--preview",
    "-pv",
    default="",
    type=str,
    help=""
)
@click.option(
    "--log_path",
    "-lp",
    default="",
    type=str,
    help=""
)
@click.pass_context
#rrrrrrrrrrr
def run(ctx, conf_path, base_conf, experiment, 
        exclude_conf, include_conf, overwrite_conf, var, 
        save_model, addto, rem, save_data, load_data, add_prefix, wrap, 
        only_var, sep, num_exps, one, cpu, undone, repeat, log_path, 
        dpy, port, break_point, reval_bests, trial, preview):

     if dpy:
        debugpy.listen(('0.0.0.0', int(port)))
        print("Waiting for client at run...port:", port)
        debugpy.wait_for_client()  # blocks execution until client is attached
     if colab and cpu:
         ans = input("Are you sure you want to use CPU?")
         if ans == "n":
             return
     if not conf_path:
        conf_path = "confs"
        if colab: conf_path = "colab_confs"
     if ctx.invoked_subcommand is None:
        mlog.info("Reading from conf %s", conf_path)
        _path = os.path.join(confPath, conf_path, experiment)
        pp = Path(__file__).parent.resolve()
        if not Path(_path).exists():
           conf = os.path.join(pp, confPath, base_conf + ".json") # default conf
           mlog.info("NEW experiment! Reading from conf %s", conf)
           if Path(conf).exists():
               with open(conf, 'r') as f:
                   args = json.load(f) 
           else:
               mlog.info(f"%s doesn't exists ...", conf)
               return
           args["config"] = ""
           args["output_name"] = "" 
           args["experiment"] = experiment 
           args["preview"] = preview 
           args["skip"] = True # skip experiment
           global logPath
           if log_path:
               logPath = log_path
           if addto:
               spath = os.path.join(logPath, addto)
           else:
               spath = os.path.join(logPath, experiment)
           if Path(spath).exists() and rem:
               #if input("Are you sure you want to delete the experiment folder?") == "y":
               #shutil.rmtree(spath)
               spath = spath.rstrip("/")
               dirs = glob.glob(spath + '/*/')
               for d in dirs:
                    shutil.rmtree(d)

           if Path(spath).is_file():
               os.remove(spath)
           Path(spath).mkdir(exist_ok=True, parents=True)
           args["save_path"] = spath
           args["load_path"] = pretPath 
           args["trial"] = trial
           _extra = ""
           exclude_list = ["no_confirm", "follow_method", "method", "test_samples"]
           mlog.info("Extra args=%s", ctx.args)
           run_args = {}
           for _item in ctx.args:
                mlog.info("arg = %s", _item)
                _key,_val = _item.split("=")
                _val = _val.strip()
                _key=_key.strip("--")
                if not _key in exclude_list:
                    _ks = "".join([k[0] for k in _key.split("_")])
                    _extra += "@" + (_ks + "=" + _val if not str(_val)=="True" else _key)
                mlog.info("set %s = %s", _key, _val)

                if _val == "null": 
                    args[_key]= ""

                if not _val == "False":
                    if _val == "True":
                        args[_key]= True
                    else:
                        args[_key]= _val
                    run_args[_key] = args[_key]
                else:
                   if _key in args:
                       del args[_key]
                   run_args[_key] = "False"
           # oooooooooooooo
           multi_only = False
           if not var:
               output_name = base_conf + sep + args["method"] + sep + _extra
               args["overwrite"] = output_name
               args["no_save_model"] = not save_model
               ctx.invoke(train, **args)
           else:
               output_name = "trial=" + args["trial"]
               var = var.replace("all_rels","#".join(all_rels))
               var = var.replace("x_rels","#".join(x_rels))
               all_vars = var.split("--")
               #all_vars = sorted(all_vars)
               var_names = [x.split("=")[0] for x in all_vars]
               values = [x.split("=")[1].split("#") for x in all_vars]
               if "rel_filter" in var_names:
                   index = var_names.index("rel_filter")
                   if "multi" in values[index] or "multi-only" in values[index]:
                       save_data = True
                   if "multi-only" in values[index]:
                       multi_only = True

               if not args["tag"]:
                   tags = []
                   for vv, cc in zip(var_names, values):
                       if len(cc) > 1:
                           tags.append(vv)
                           if preview and not vv in preview:
                               var_names.remove(vv)
                               values.remove(cc)

                   args["tag"] = "@".join(tags)
               tot_comb = [dict(zip(var_names, comb)) for comb in itertools.product(*values)]
               ii = 0
               orig_args = args.copy()
               inp_test_samples = args["test_samples"]
               if one: num_exps = 1
               if num_exps > 0:
                   tot_comb = tot_comb[:num_exps]
               mlog.info("Total experiments:%s", len(tot_comb))
               for comb in tot_comb:
                   _output_name = output_name
                   __output_name = output_name
                   for var_name,var_item in comb.items():
                       if var_item == "null": var_item = ""
                       if var_item.strip() == "False": 
                           if var_name in args:
                               del args[var_name]
                           run_args[var_name]= "False"
                       else:
                           if var_item.strip() == "True":
                               var_item = True
                           args[var_name]=var_item
                           orig_args[var_name] = var_item
                           run_args[var_name] = var_item
                       if not var_name in exclude_list:
                           _output_name +=  sep + var_name + "=" + str(var_item)
                       __output_name +=  sep + var_name + "=" + str(var_item)
                   ii += 1
                   rel_folder = "all" if not args["rel_filter"] else args["rel_filter"]
                   if only_var:
                       args["overwrite"] = __output_name.strip(sep)
                   else:
                       args["overwrite"] = args["method"] + sep + rel_folder + sep + _output_name \
                           + sep + _extra 
                   args["no_save_model"] = not save_model
                   if load_data:
                       args["data_path"] = spath
                       args["use_all_data"] = True
                   if undone:
                       args["undone"] = True
                   if repeat:
                       args["skip"] = False
                   if preview:
                       break_point = "data"
                   args["break_point"] = break_point 
                   mylogs.BREAK_POINT = break_point
                   if cpu:
                       args["cpu"] = True
                       os.environ["CUDA_VISIBLE_DEVICES"] = ""
                   if add_prefix:
                       args["tag"] = "experiment"
                   if wrap:
                       args["method"] += "-wrap"
                   if save_data: 
                       args["save_data"] = spath
                   if "multi" in args["rel_filter"]:
                       mbp("multi")
                       args["data_path"] = spath
                       args["rel_filter"] = "" 
                       args["multi"]= True
                       args["use_all_data"] = True
                       args["save_data"] = False
                   else:
                       if multi_only: 
                           args["preview"] = "multi_only"
                           args["save_data"] = spath
                       _dp = os.path.join(dataPath,"sel",args["rel_filter"] + ".tsv")
                       if not load_data:
                           args["data_path"] = orig_args["data_path"]
                       args["multi"] = False 
                       if Path(_dp).is_file():
                           args["test_samples"] = 0
                           args["test_path"] = _dp
                       elif not load_data:
                           args["use_all_data"] = False
                           args["test_path"] = orig_args["test_path"]
                           args["test_samples"] = orig_args["test_samples"] 
                   args["exp_id"] = ii
                   ow = args["overwrite"]
                   if reval_bests:
                       lp = os.path.join(spath, ow, "best_model")
                       if not Path(lp).exists():
                           mlog.info("Skipping reval, no saved model")
                           continue
                       else:
                           mlog.info("Loading model from %s", lp)
                           args["do_eval"] = True
                           args["test_set"] = "validation" 
                           args["load_path"] = lp
                           args["trial"] = "reval"
                   ctx.invoke(train, **args, run_args = run_args)
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
    "--exp_id",
    "-id",
    default=1,
    type=int,
    help="A number for experiment showing its order"
)
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
    default=10,
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
    "--pl_learning_rate",
    "-plr",
    default=0,
    type=float,
    help="prompt tuning learning rate"
)
@click.option(
    "--router_lr",
    "-rlr",
    default=0.0005,
    type=float,
    help="prompt tuning router learning rate"
)
@click.option(
    "--learning_rate",
    "-lr",
    default=0,
    type=float,
    help="fine tuning learning rate"
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
    "--freeze_step",
    "-fs",
    default=0,
    type=int,
    help=""
)
@click.option(
    "--unfreeze_step",
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
    default="train.tsv",
    type=str,
    help=""
)
@click.option(
    "--val_path",
    "-vp",
    default="valid.tsv",
    type=str,
    help=""
)
@click.option(
    "--test_path",
    "-tep",
    default="test.tsv",
    type=str,
    help=""
)
@click.option(
    "--sample_path",
    "-samp",
    default="sample.tsv",
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
    default=30,
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
    default="",
    type=str,
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
    default="top_k",
    type=str,
    help=""
)
@click.option(
    "--wb",
    "-wb",
    is_flag=True,
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
    "--log_per_exp",
    "-lfe",
    default="",
    type=str,
    help="save disticnt log for experiments in output (good for comparison)"
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
    "-perrecs",
    is_flag=True,
    help="Show if train_samples are records or unique heads"
)
@click.option(
    "--per_prefix",
    "-perpre",
    is_flag=True,
    help="Show if train_samples are per relations (prefix) or all the samples"
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
    "--group_sets",
    "-gs",
    default="",
    type=str,
    help="The name of splits to group by columns specified in group_by (below)"
)
@click.option(
    "--group_by",
    "-gb",
    default="prefix@input_text",
    type=str,
    help="The column name to do group_by on them"
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
    "--flat_prompts",
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
    default="rouge",
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
    "--no_save_best",
    "-nsb",
    is_flag=True,
    help=""
)
@click.option(
    "--gen_bs",
    "-gb",
    default="",
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
    "--unfreeze_parts",
    "-ufzl",
    default="",
    type=str,
    help="Layers to be freezed"
)
@click.option(
    "--freeze_parts",
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
    default="",
    type=str,
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
    default="custom",
    type=str,
    help="training loop"
)
@click.option(
    "--know",
    "-ku",
    default="s",
    type=str,
    help=""
)
@click.option(
    "--preview",
    "-pv",
    default="",
    type=str,
    help="Don't run the trainer and just show initial settings"
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
    "--tag",
    "-tag",
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
@click.option(
    "--use_all_data",
    "-uad",
    is_flag=True,
    help=""
)
@click.option(
    "--multi",
    "-multi",
    is_flag=True,
    help="A tag indicating multi-task"
)
@click.option(
    "--temp_num",
    "-tid",
    default=1,
    type=int,
    help="The number of template for each relation"
)
@click.option(
    "--undone",
    "-ud",
    is_flag=True,
    help="Only shows the path without running..."
)
@click.option(
    "--someone",
    "-so",
    is_flag=True,
    help=""
)
@click.option(
    "--run_args",
    "-ra",
    default={},
    type=dict,
    help=""
)
@click.option(
    "--match",
    "-match",
    default="",
    type=str,
    help=""
)
@click.option(
    "--dpy",
    "-dpy",
    is_flag=True,
    help="Enables remote debugging"
)
@click.option(
    "--prompt_tune",
    "-ptune",
    is_flag=True,
    help=""
)
@click.option(
    "--prompt_config_file",
    "-pcf",
    default="",
    type=str,
    help=""
)
@click.option(
    "--load_prompt",
    "-lpropmpt",
    default="",
    type=str,
    help=""
)
@click.option(
    "--data_name",
    "-dn",
    default="",
    type=str,
    help=""
)
@click.option(
    "--seed",
    "-seed",
    default=0,
    type=int,
    help=""
)
@click.option(
    "--do_valid",
    "-do_valid",
    is_flag=True,
    help=""
)
@click.option(
    "--break_point",
    "-stbr",
    default=0,
    type=int,
    help=""
)
@click.option(
    "--skilled_variant",
    "-sv",
    default="",
    type=str,
    help=""
)
@click.option(
    "--int_dim",
    default=300,
    type=int,
    help=""
)
@click.option(
    "--init_temperature",
    default=1.,
    type=float,
    help=""
)
@click.option(
    "--prompt_token_num",
    default=10,
    type=int,
    help=""
)
@click.option(
    "--n_skills",
    default=0,
    type=int,
    help=""
)
@click.option(
    "--n_prompts",
    default=8,
    type=int,
    help=""
)
@click.option(
    "--trunc_router",
    "-tr",
    default="none",
    type=str,
    help="Trunc router in wrapper"
)
@click.option(
    "--general_type",
    "-gt",
    default="lstm",
    type=str,
    help="The type of general prompt encoders"
)
@click.option(
    "--router_variant",
    "-rv",
    default="fixed",
    type=str,
    help=""
)
@click.option(
    "--freeze_target",
    "-ft",
    default="model",
    type=str,
    help="The target for freeze and unfreeze"
)
@click.option(
    "--freeze_skill",
    "-fsk",
    is_flag=True,
    help=""
)
@click.option(
    "--add_prior",
    "-apr",
    is_flag=True,
    help=""
)
@click.option(
    "--freeze_exclude",
    "-",
    default="",
    type=str,
    help=""
)
@click.option(
    "--config_file",
    "-conf",
    default="",
    type=str,
    help=""
)
@click.option(
    "--stype",
    "-stype",
    default="",
    type=str,
    help="sub type for model"
)
def train(exp_id, model_id, experiment, qtemp, anstemp, extemp, method, val_method, train_samples, test_set, val_samples, test_samples, load_path, data_path, train_path, val_path, test_path, sample_path, overwrite, save_path, output_name, lang, pred_tresh, ignore_blanks,only_blanks, include, exclude, nli_group, learning_rate, pl_learning_rate, router_lr, do_eval, cont, wrap, prefix, frozen, freeze_step, unfreeze_step, cpu, load_prompt_path, verbose, cycle, batch_size, path, from_dir, is_flax, config,clear_logs, gen_param, print_log, log_per_exp, wb, training_round, epochs_num, per_record, per_prefix, is_even, start, prompt_length, prompt_pos, zero_shot, sampling, opt_type, samples_per_head, group_sets, group_by, deep_log, trans, encoder_type, rel_filter, ex_type, last_data, save_df, flat_prompts, num_workers, scorers, train_start, no_save_model, no_save_best, gen_bs, shared_embs, no_confirm, follow_method, repeat, trial, unfreeze_parts, freeze_parts, pid, use_dif_templates, break_sent,sort, do_preproc, replace_blanks, loop, know, preview, ph_num, save_data, tag, skip, use_all_data, multi, temp_num, undone, someone, run_args, match, dpy, prompt_tune, prompt_config_file, load_prompt, data_name, seed, do_valid, break_point, skilled_variant, int_dim, prompt_token_num, n_skills, n_prompts, init_temperature, trunc_router, general_type, router_variant, freeze_target, freeze_skill, add_prior, freeze_exclude, config_file, stype):

    #%% some hyper-parameters

    mylogs.BREAK_POINT = break_point
# Allow other computers to attach to debugpy at this IP address and port.
    if dpy:
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for client... at train")
        debugpy.wait_for_client()  # blocks execution until client is attached
    #bbbbbbbbbbb
    mbp("start")
    #underlying_model_name = "logs/atomic-mt5/last"
    vlog.info("given load path: %s", load_path)
    vlog.info("given load path: %s", load_path)
    vlog.info("given save path: %s", save_path)

    if int(test_samples) < 0:
        test_set = "" 

    seed = int(seed)
    set_random_seed(seed)
    
    args = locals() #run_args # input parameters
    set_args(args.copy())
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,
                               AdapterTrainingArguments))
    model_args = data_args = training_args = adapter_args = None
    if config_file and config_file.endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=config_file)

    if wb:
        wandb.init(project="plearning")

    if "dlog" in print_log: # data logger
        dlog.addHandler(consoleHandler)
        dlog.setLevel(logging.DEBUG)
    if "vlog" in print_log: # evaluation logger
        vlog.addHandler(consoleHandler)
        vlog.setLevel(logging.DEBUG)
    if "clog" in print_log: # config logger
        clog.addHandler(consoleHandler)
        clog.setLevel(logging.DEBUG)


    if log_per_exp:
        for logger, fname in zip([mlog,dlog,clog,vlog,ttlog,timelog], ["main","data","cfg","eval","train", "time"]):
            logger.setLevel(logging.INFO)
            logFilename = os.path.join("output", str(exp_id) + "_" + fname + ".log")
            handler = logging.FileHandler(logFilename, mode="w")
            handler.setFormatter(FORMAT)
            logger.handlers.clear()
            logger.addHandler(handler)
    mlog.info(f"========================= {experiment}:{exp_id} ========================")
    if save_path == "":
        if "ahmad" or "pouramini" in home:
            save_path = os.path.join(home, "logs")
        else:
            save_path = "/content/drive/MyDrive/pouramini/logs"

    if freeze_parts == "none":
        frozen = False
        freeze_parts = ""

    if "-wrap" in method and not wrap:
        mlog.info("Method %s is for wrapped models...", method)
        wrap = True
        if wrap and not frozen and follow_method and not "-nfz" in method:
            frozen = True
    if wrap and not frozen and not "-nfz" in method:
         if not no_confirm:
             ans = input("Are you sure you want to wrap without freezing the model?")
             if ans != "y":
                 frozen = True
    method = method.replace("-nfz", "")
    w_str = "wrapped" if wrap else "unwrapped"
    f_str = "freezed" if frozen else "unfreezed"
    if not output_name and not (cont or do_eval):
        output_name = model_id + "_" + method 
    conf_path = save_path
    if model_id == "test":
        save_path = ""
        output_name = "test"
    if config:
        conf_path = confPath 
        Path(conf_path).mkdir(exist_ok=True, parents=True)
        args["config"] = ""
        args["output_name"] = ""
        del args["run_args"]
        with open(os.path.join(conf_path, f'{config}.json'), 'w') as outfile:
            json.dump(args, outfile, indent=4)
        mlog.info("Config %s was created at %s %s", conf_path, config + ".json", conf_path)
        return
    if not data_name:
        if not data_path:
            data_path = dataPath
            train_path = os.path.join(data_path, train_path) 
            test_path = os.path.join(data_path, test_path)
            val_path = os.path.join(data_path, val_path) 
            sample_path = os.path.join(data_path, sample_path) 
        else:
            train_path = os.path.join(data_path, "train.tsv")
            test_path = os.path.join(data_path, "test.tsv")
            val_path = os.path.join(data_path, "valid.tsv")
            sample_path = os.path.join(data_path, "sample.tsv")

        assert Path(train_path).is_file(), f"Train path {train_path} is not!"

    if use_all_data:
        train_samples, test_samples = 0, 0

    if not load_path:
        load_path = os.path.join(home, "pret")

    if from_dir:
        underlying_model_name = path
    elif Path(load_path).exists():
        bins = glob.glob(f"{load_path}/*.bin")
        if not bins and do_eval:
            print("Skipping.... the folder contains no model (*.bin) !!")
            return
        underlying_model_name = f"{load_path}/{model_id}" if not bins else load_path
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
    pl_learning_rate = float(pl_learning_rate)
    lr_mt = {}
    if learning_rate == 0: 
        if opt_type == "adam": 
            learning_rate = lr_mt[method] if method in lr_mt else 3.25e-04  
        else:
            learning_rate = 1e-3
        if "gpt" in model_id:
            learning_rate = 1e-5
    if pl_learning_rate == 0: 
        if encoder_type == "lstm":
            pl_learning_rate = 0.03  
        elif encoder_type == "emb":
            pl_learning_rate = 0.1  
        else:
            pl_learning_rate = 0.01  

    assert learning_rate > 0, "Learning rate is zero!"
    assert pl_learning_rate > 0, "Prompt tuning Learning rate is zero!"
    device = 'cuda' if not cpu and torch.cuda.is_available() else 'cpu'

    log_dir = save_path
    set_device(device)
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
    if Path(save_path).exists() and skip and not do_eval:
        tsv_files = glob.glob(save_path + "/**/*.tsv", recursive = True)
        if tsv_files:
            print(save_path + ") Skipping.... the folder already exists!!")
            return
    if undone: # only report it wasn't done
        _ss = save_path.split("/")
        for _mm in _ss:
            if "=" in _mm:
                print(_mm)
        return

    mlog.info("Optimizer type %s:", opt_type)
    mlog.info("learning rate %s:", learning_rate)
    mlog.info("pl learning rate %s:", pl_learning_rate)
    mlog.info("Output name: %s", output_name)
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
    for logger, fname in zip([mlog,dlog,clog,vlog,ttlog], ["main","data","cfg","eval","train"]):
        if len(logger.handlers) >= 3:
            continue
        logger.setLevel(logging.INFO)
        logFilename = os.path.join(save_path, fname + "_.log")
        handler = logging.FileHandler(logFilename, mode = "w" if clear_logs else "a")
        logger.addHandler(handler)

    for logger in [mlog, clog, dlog, ttlog, vlog]:
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
        config = {} 
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
            if stype == "atm":
                config = tp.T5Config.from_pretrained(
                    underlying_model_name
                    #cache_dir=model_args.cache_dir,
                    #revision=model_args.model_revision,
                    #use_auth_token=True if model_args.use_auth_token else None,
                )
                config.train_task_adapters = adapter_args.train_task_adapters
                config.prefix_tuning = adapter_args.prefix_tuning
                config.attn_prefix_tuning = model_args.attn_prefix_tuning
                config.attn_method = model_args.attn_method
                config.ignore_target = model_args.ignore_target
                config.shared_attn = model_args.shared_attn
                config.prefix_num = model_args.prefix_num
                config.num_target = len(data_args.task_name)
                config.temperature = model_args.temperature
                config.learned_temperature = model_args.learned_temperature
                config.fix_attention = model_args.fix_attention
                adapter_config = get_adapter_config(
                    adapter_args, data_args, training_args, config)

                model = tp.T5ForConditionalGeneration.from_pretrained(
                                                         underlying_model_name, 
                                                         #output_attentions = False, 
                                                         config = config,
                                                         adapter_config=adapter_config,
                                           # Whether the model returns attentions weights.
                                                         #output_hidden_states = False,
                                                         #return_dict=True
                                                           ) 
                ##################################################
                if model_args.load_prefix_embeddings is True:
                    if model_args.prompt_embedding_path is None:
                        for name, param in model.named_parameters():
                            if "prefix_shared" in name or "prefix" in name:
                                shared_params = [param]
                    else:
                        shared_params = []
                        mapl=torch.device('cpu')
                        for path in model_args.prompt_embedding_path:
                            shared_param = torch.load(path, map_location=mapl)
                            shared_params.append(shared_param)
                        if model_args.target_prompt_embedding_path is not None:
                            target_prompt_embedding = torch.load(
                                model_args.target_prompt_embedding_path,
                                map_location=mapl)

                    if model_args.attn_prefix_tuning is True:
                        if training_args.do_train is True and model_args.multi_task is False and model_args.shared_attn is False:
                            # Initialize the prompt embeddings using the first prompts
                            # Load all of the target prompts
                            model.store_prefix_weights(shared_params)
                            model.update_prefix_weights_single(shared_params[0])
                        elif training_args.do_train is True and model_args.multi_task is False and model_args.shared_attn is True:
                            # initialize the embeddings
                            # initialize multiple shared embeddings
                            model.store_prefix_weights(shared_params)
                            model.update_prefix_weights_multi(
                                shared_params[0], num_target=config.num_target)
                        else:
                            # Load prompt embeddings except for the last one
                            # Load last prompt embeddings to initialize the target prompt embeddings.
                            model.store_prefix_weights(shared_params)
                            model.update_prefix_weights_single(shared_params[-1])

                    else:
                        if model_args.target_prompt_embedding_path is None:
                            model.update_prefix_weights(shared_params)
                        else:
                            model.update_prefix_weights(
                                shared_params, target_prompt_embedding)

                if model_args.load_attention is True and model_args.attn_path is not None:
                    model.update_attention_weights(torch.load(model_args.attn_path), 
                            map_location=mapl)

                if model_args.load_attention is True and model_args.attn_path_sub is not None:
                    model.update_attention_weights_sub(model_args.attn_path_sub)

                if model_args.load_layer_norm is True and model_args.layer_norm_dir is not None:
                    model.update_layer_norm_weights(model_args.layer_norm_dir)

            #################################################
            else:
                model = T5ForConditionalGeneration.from_pretrained(
                                                         underlying_model_name, 
                                                         #output_attentions = False, 
                                           # Whether the model returns attentions weights.
                                                         #output_hidden_states = False,
                                                         #return_dict=True
                                                           ) 
            tokenizer = AutoTokenizer.from_pretrained(tpath)

        if underlying_model_name == model_id:
            mlog.info("Saving model on local %s", load_path)
            model.save_pretrained(os.path.join(load_path, model_id))
            tokenizer.save_pretrained(os.path.join(load_path, model_id))
            underlying_model_name = os.path.join(load_path, model_id)
        return model, tokenizer, underlying_model_name, config

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
        model, tokenizer, underlying_model_name, atm_config = load_model(model_id, underlying_model_name)
        for split_name, df in atomic_dataset.items():
            mlog.info("Translating ...%s ", split_name)
            if trans != split_name:
                continue
            path = train_path if split_name == "train" else val_path
            model.to(device=device)
            logger = ttlog 
            mlog.info("Translating %s", path)
            trans_df = translate(model, tokenizer, df, "target_text@fa@5000", path, logger, start, load_path) 
            translate(model, tokenizer, trans_df, "input_text@fa@5000", path, logger, start, load_path) 
        return
    
    mlog.info("Loading from %s", underlying_model_name)
    model, tokenizer, underlying_model_name, atm_config = load_model(model_id, underlying_model_name)
    extend_tokenizer(tokenizer)
    if prompt_length:
        length = [int(s) for s in str(prompt_length).split("-")]
        set_prompt_lengths(rel_filter, length)

    sample_samples = 0
    num_samples = {"train": int(train_samples), "validation":int(val_samples), "test":int(test_samples), "sample":int(sample_samples)}
    split_path = {"train":train_path, "validation":val_path, "test":test_path, "sample":sample_path}
    save_ds_path = {}
    for split, _path in split_path.items():
        #_path = _path.replace(".tsv","_")
        save_ds_path[split] = os.path.join(underlying_model_name, split)
    #tokenize_relations(tokenizer)
    atomic_query_responses = {"train":[], "validation":[]}
    generated_samples = {}
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
        # Reset some global variables
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
            group_them = []
            if split_name in group_sets:
                #tails_per_head = 3
                #TODO why group them?
                group_them = group_by
            #    _replace_blanks = False
            df_path = split_path[split_name]
            if split_name == "sample" and not Path(df_path).is_file():
                mlog.info("No sample was provided! %s ", df_path)
                continue
            split_df = pd.read_table(df_path)
            mlog.info("Path of dataset for %s %s", split_name, split_path[split_name])
            dlog.info("Columns of %s\n  %s", split_name, "\n".join(list(split_df.columns)))
            dlog.info("Len of %s\n  %s", split_name, len(split_df))
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
            _match = match
            match_split = "train"
            if "@" in _match:
                _match, match_split = match.split("@")
            _match = _match if _match != "none" else ""
            if match_split != "both" and split_name != match_split:
                _match = ""
            if "test" in split_name:
                _method = val_method
                #_only_blanks = True
            _repeat = 1
            _break_sent = break_sent if break_sent != "none" else ""
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
                                pred_tresh, nli_group, 
                                per_record, per_prefix, is_even, start, 
                                sampling, ex_type,
                                tails_per_head, save_ds_path[split_name], _repeat, 
                                int(pid), _break_sent, sort, _replace_blanks, 
                                None, int(ph_num), group_them = group_them, 
                                temp_num = temp_num, someone=someone, 
                                match=_match, batch_size=int(batch_size),
                                freeze_step= int(freeze_step),
                                unfreeze_step = int(unfreeze_step)
                        )
            if save_data:
                myds[_name].save_data(os.path.join(save_data,_name + ".tsv"), merge=True)
                if _name == "train":
                    myds[_name].save_data(os.path.join(save_data, "sample.tsv"), merge=True, sample=10)
        return myds

    if do_eval:
        myds = load_data([test_set])
        val_records = myds[test_set].num_records
        train_records = 0
    elif not data_name:
        ds_list = ["train"]
        if do_valid:
            ds_list += ["validation"]
        ds_list += ["sample"]
        myds = load_data(ds_list)
        example = ""
        sample_dataset = None
        if True: #wrap or preview: It's required to retrive prompt tokens
            if "sample" in myds:
                mbp("sample")
                samples_iter = iter(myds["sample"])
                sample_limit = myds["sample"].num_records
                sample_dataset= myds["sample"]
            else:
                samples_iter = iter(myds["train"])
                sample_limit =  int(batch_size)
                sample_dataset= myds["train"]
            ii = 0
            _sample = True
            generated_samples["sample"] = []
            logger = mlog
            logger.info("----------- SAMPLES -------------")
            logger.info(f"----------- {method} -------------")
            logger.info("----------- SAMPLES -------------")
            while _sample and ii < sample_limit: 
                logger.info(f"----------- {method} -------------")
                _sample = next(samples_iter, None)
                if not example:
                    example = _sample
                if not _sample:
                    break
                if _sample["rep"] > 0:
                    continue
                ii += 1
                logger.info(_sample)
                if False: #_sample:
                    generated_samples["sample"].append((_sample[0], _sample[1]))
            logger.info("--------------------------------")
            logger.info("Preparing samples: %s ", len(generated_samples["sample"]))
    extend_tokenizer(tokenizer)
    prompt_config = None
    if prompt_tune:
        with open(prompt_config_file, "r") as f:
            prompt_config = json.load(f)
            if load_prompt is not None:
                prompt_config["load_prompt"] = load_prompt
            for t in ["enc", "dec"]:
                prompt_config[t]["init_ids"] = tokenizer.encode(prompt_config[t]["init_tokens"])
                pad_num = prompt_config[t]["prompt_len"] - len(prompt_config[t]["init_ids"])
                prompt_config[t]["init_ids"].extend(tokenizer.convert_tokens_to_ids([prompt_config[t]["default_init_token"] for _ in range(pad_num)]))
                prompt_config[t]["init_ids"] = torch.tensor(prompt_config[t]["init_ids"], dtype=torch.long).to(device)


    def load_data2(data_path, data_type, tokenizer, prompt_config=None, ratio=1, num=-1, drop_last=True, do_infer=False):
        data_path = os.path.join(data_path, data_type + ".jsonl") 

        # Data parallel arguments.
        #debugpy.breakpoint()  # or debugpy.breakpoint()
        world_size = 1 
        rank = 0 
        args = dotdict({})
        args.log_file = os.path.join(logPath, "ppt.log")
        args.batch_size = batch_size
        args.dev_batch_size = batch_size*2
        args.eval_batch_size = batch_size*2
        if data_type == "train":
            global_batch_size = args.batch_size * world_size
        elif data_type == "valid":
            global_batch_size = args.dev_batch_size * world_size
        else:
            global_batch_size = args.eval_batch_size * world_size

        dataset = DATA_CONFIG[data_name]["dataset"](
            args,
            tokenizer,
            data_path,
            data_type,
            ratio=ratio,
            num=num,
            prefix=args.data_prefix,
            do_infer=do_infer,
            prompt_config=prompt_config)

        if data_type == "train":
            sampler = RandomSampler(dataset)
            sampler.set_seed(seed)
        else:
            sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler=sampler,
                                    batch_size=global_batch_size,
                                    drop_last=drop_last)

        data_loader = DataLoader(dataset,
                                 batch_sampler=batch_sampler,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 collate_fn=dataset.collate)

        # Torch dataloader.
        return data_loader, dataset, sampler

    modules_to_freeze = []
    modules_to_unfreeze = []
    if "model@" in freeze_parts:
        _parts = freeze_parts.split("@")
        enc_parts = _parts[1]
        freeze_self_att(modules_to_freeze, enc_parts, model.encoder)
        if len(_parts) == 3:
            dec_parts = _parts[2]
            freeze_self_att(modules_to_freeze, dec_parts, model.decoder, True)
        if len(_parts) == 4:
            decx_parts = _parts[3]
            freeze_cross_att(modules_to_freeze, decx_parts, model.decoder)

    def freeze(modules_to_freeze, exclude=""):
        for module in modules_to_freeze:
            if hasattr(module, "parameters"):
                for param in module.parameters():
                    if exclude and exclude in name:
                        continue
                    param.requires_grad = False  # Actual freezing operation
            else:
                module.requires_grad = False  # Actual freezing operation

    def unfreeze(modules_to_unfreeze):
        for module in modules_to_unfreeze:
            if hasattr(module, "parameters"):
                for param in module.parameters():
                    param.requires_grad = True  # Actual freezing operation
            else:
                module.requires_grad = True  # Actual freezing operation


    mlog.info("len tokenizer %s", len(tokenizer))
    my_specials = [x for x in tokenizer.additional_special_tokens if not "<extra_id"  in x]
    vlog.info("list of spcial tokens: %s", my_specials)
    extra = "_" + now
    m_name = model_id + "-" + method
    p_str = "prefixed" if prefix else "not_prefixed"
    _tag = ""
    taginfo = ""
    for _t in tag.split("@"):
        taginfo += "|" + _t  
        if _t in args:
            _tag += "|" + str(args[_t])
        else:
            _tag += "|" + _t  
    tag = _tag.strip("|")
    exp_info = {"exp":experiment + "-" + str(exp_id), 
                "expid": str(exp_id),
                    "model":model_id, 
                    "lang": lang, 
                    "method":method, 
                    "wrap": w_str + ("-" + encoder_type if wrap else ""),
                    "frozen":f_str, 
                    "prefixed":p_str,
                    "pid":pid,
                    "tag":tag,
                    "taginfo":taginfo,
                    "n_prompts":n_prompts,
                    "multi":multi, 
                    "steps":str(train_samples)+"x"+str(repeat)+"x"+str(epochs_num),
                    "plen":prompt_length,
                    "opt_type":opt_type,
                    "trial":trial,
                    "exp_trial":str(experiment) + "-" + str(exp_id)+ "-" + str(trial),
                    "learning_rate":learning_rate,
                    "pl_learning_rate":pl_learning_rate,
                    "date":extra}
    exp_info["eval"] = do_eval
    for k,v in run_args.items():
        if not k in exp_info:
            exp_info[k] = v

    mbp("start")
    if not gen_bs:
        gen_bs = str(batch_size) + "@" + str(batch_size)
    def compute_metrics(eval_preds):
        preds, labels, data_info = eval_preds
        #post_processor = AutoPostProcessor.get(data_args.dataset_name[0], tokenizer,
        #                                       data_args.ignore_pad_token_for_loss)
        #decoded_preds, decoded_labels = post_processor.process(
        #    preds, labels, data_info)
        result = {}
        eval_metrics = [metrics.rouge]
        for metric in eval_metrics:
            result.update(metric(preds, labels))
        return result

    def eval_test(model, tokenizer, result_fname=""):
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
            exp_info["val_records"] = val_records 
            mbp("start")
            #a1, a2, s1, r = evaluate1(tokenizer, test_dataloader, model, device, seed, mode="test", save_path=save_path, wrap=False, task_ids=task_ids)
            #mlog.info("acc1: %s, acc2: %s, sts: %s, res: %s", a1, a2, s1, r)
            mbp("start")
            mbp(2)
            _model = wrapped_model
            if task_ids is not None:
                _model = wrapped_model 
            if not result_fname:
                _save_path = save_path
            else:
                _save_path = os.path.join(save_path, result_fname)
            evaluate(test_dataset, test_dataloader, _save_path, exp_info, val_records, gen_param, scorers = scorers, batch_size=gen_bs, model=_model, tokenizer=tokenizer, set_name=_set, seed=seed, task_ids=task_ids)  
    if do_eval or (not wrap and frozen and modules_to_freeze is model):
        mlog.info("Evaluating the model...")
        model.to(device=device)
        eval_dataset = myds[test_set]#.map(tokenize)
        data_collator = MyCollator(tokenizer, model, ds_type=test_set, prefix=prefix)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
            batch_size=batch_size,shuffle=False,
            collate_fn=data_collator,
        )
        a1, a2, s1, eval_loss = evaluate1(tokenizer, eval_dataloader, model, device, seed, mode="dev", save_path=save_path, wrap=False)
        log_string = "eval_loss: " + str(eval_loss) + " | dev acc({}, {} st:{}): ".format(a1, a2, s1) 
        mlog.info(log_string)
        eval_test(model, tokenizer, "reval_full.tsv")
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

    accumulation_tiny_steps = 1 
    batch_size = int(batch_size)
    if "gpt" in model_id:
        accumulation_tiny_steps = 1
    if batch_size >= 2:
        node_batch_size = batch_size//accumulation_tiny_steps
    else:
        accumulation_tiny_steps = 1
        node_batch_size = 1

    train_dataset = myds["train"]#.map(tokenize)
    iterations = train_dataset.num_records//batch_size
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
    wrapped_model = None

    mbp("start")
    if not load_prompt_path and Path(os.path.join(load_path, model_id, "prompt")).exists():
        load_prompt_path = os.path.join(load_path, model_id, "prompt")
        mlog.info("prompt path:%s ", load_prompt_path)
    mlog.info("Wrapping the model ...")
    model_to_wrap = model
    if prefix:
        model_to_wrap = model.pretrain_model

    task_ids = None
    n_prompts = int(n_prompts)
    n_tasks = len(sample_dataset.tasks)
    n_skills = int(n_skills)
    if skilled_variant == "private":
        n_skills = n_tasks
    if skilled_variant == "shared":
        n_skills = 1
    prompt_token_num = int(prompt_token_num)
    if skilled_variant == "none":
        skilled_variant = ""
    if skilled_variant:
       task_ids = torch.LongTensor(range(n_tasks))
    prefix_config = {
        'intrinsic_dim': int_dim,
        'n_prompt_tokens': prompt_token_num,
        'n_tasks': n_tasks,
        'n_prompts': n_prompts,
        'temperature': init_temperature,
    }
    #prefix_config = None

    general_prompts = {}
    prompts = {} 
    if not skilled_variant:
        for n in range(prompt_token_num):
            l = []
            for m in range(prompt_token_num):
                l.append("<g"+str(n) + "@" + general_type + "_" + str(m)+ ">") 
            general_prompts["g"+str(n) + "@" + general_type]  = l 
        if flat_prompts == "none": flat_prompts = ""
        assert flat_prompts != "none"
        prompts = sample_dataset.prompts
    wrapped_model = model_to_wrap
    if adapter_args and not adapter_args.prefix_tuning:
        wrapped_model = wrap_model(model_to_wrap, tokenizer, encoder_type, load_prompt_path, flat_prompts=flat_prompts, method = method, shared_embs= shared_embs, skilled_variant=skilled_variant, prefix_config=prefix_config, n_tasks=n_tasks, n_skills=n_skills, 
            exp_id=exp_id, 
            encoder_prompts= prompts,
            general_prompts= general_prompts, 
            router_variant=router_variant, device=device) 

    if not prefix:
        model.resize_token_embeddings(len(tokenizer))
    else:
        model.pretrain_model.resize_token_embeddings(len(tokenizer))

    def add_parts(_list, parts):
        if parts == "encoder":
            for encoder in wrapped_model.general_encoders:
                _list.append(encoder)
            for encoder in wrapped_model.prompt_encoders:
                _list.append(encoder)
        if parts == "router":
            for encoder in wrapped_model.general_encoders:
                if encoder.router is not None: 
                    _list.append(encoder.router)
            for encoder in wrapped_model.prompt_encoders:
                if encoder.router is not None: 
                    _list.append(encoder.router)

    add_parts(modules_to_freeze, freeze_parts)
    add_parts(modules_to_unfreeze, unfreeze_parts)

    if freeze_exclude == "none":
        freeze_exclude = ""
    if frozen: # and stype != "atm":
        if stype == "atm": 
            adapter_config = get_adapter_config(
                    adapter_args, data_args, training_args, atm_config)
            model = modify_model_after_init(
                model, training_args, adapter_args, adapter_config)
        elif "model@" in freeze_parts:
            freeze(modules_to_freeze)
        else:
            if skilled_variant:
                freeze([model], exclude=freeze_exclude)
                freeze(modules_to_unfreeze)
            else:
                freeze([model])
                freeze(modules_to_unfreeze)


    fname = "output/" + str(experiment) + "-" + str(exp_id) + "-" + flat_prompts + ".txt"
    Path("output").mkdir(exist_ok = True)
    if isinstance(wrapped_model, PTuningWrapper):
        f = open(fname, "w")
        print("Number of prompts:" + str(len(wrapped_model.prompt_encoders)), file=f)
        print("Train prompts:", prompts, file=f) 
        if wrapped_model.flat_encoder:
            print(wrapped_model.flat_encoder.id_offset, file=f)
            print(wrapped_model.flat_encoder.prompt_ids, file=f)
            print(wrapped_model.flat_encoder.input_ids, file=f)
            print("Merge Encoder:", file=f)
            print(wrapped_model.flat_encoder, file=f)
        if wrapped_model.prompt_encoders:
            for encoder in wrapped_model.prompt_encoders:
                print(encoder.id_offset, file=f)
                print(encoder.prompt_ids, file=f)
                print(encoder.input_ids, file=f)
                print(encoder, file=f)
        print(model, file=f)
        jargs = json.dumps(args, indent=2)
        print(jargs, file=f)
        f.close()
    mlog.info("len tokenizer after extending %s", len(tokenizer))
    # ooooooooooooo
    if preview:
        return

    wrapped_model.to(device=device)
    if isinstance(wrapped_model, PTuningWrapper):
        if wrapped_model.prompt_encoders:
            mlog.info("Number of encoders: %s", len(wrapped_model.prompt_encoders))
            for encoder in wrapped_model.prompt_encoders:
                mlog.info("Prompt encoder %s", encoder.name)
                if encoder.router is not None: 
                    encoder.router.to(device=device)
                encoder.device = device
            for encoder in wrapped_model.general_encoders:
                mlog.info("General encoder %s", encoder.name)
                encoder.device = device
            wrapped_model.prompt_encoders.to(device=device)
            wrapped_model.general_encoders.to(device=device)

        if wrapped_model.merge_encoder:
            wrapped_model.merge_encoder.to(device)

    mlog.info("len tokenizer after wrapping %s", len(tokenizer))
    mbp("start")
    if wrapped_model:
        #model.get_input_embeddings().weight.requires_grad = False
        rgrad = [p for p in wrapped_model.parameters() if p.requires_grad]
        nrgrad = [p for p in wrapped_model.parameters() if not p.requires_grad]

    if isinstance(wrapped_model, PTuningWrapper):
        _sum = 0
        for encoder in wrapped_model.prompt_encoders:
            enc_rgrad = [p for p in encoder.parameters() if p.requires_grad]
            mlog.info("len Encoder require grad %s: %s",encoder.name, len(enc_rgrad))
            mlog.info("Encoder prompt ids: %s", encoder.prompt_ids)
            _sum += len(encoder.prompt_ids)
            mlog.info("len prompt ids %s: %s",encoder.name, len(encoder.prompt_ids))

        mlog.info("Total prompt ids: %s", _sum)
    mlog.info("Wrapped model require grad %s, ", len(rgrad))
    mlog.info("Wrapped model not require grad %s, ", len(nrgrad))
     
    mbp("wrap")

    if not no_save_model:
        tokenizer.save_pretrained(save_path)
    def get_optimizer(model, learning_rate, opt_type):
        mbp("optim")
        if model_args and model_args.attn_learning_rate is not None:
            all_parameters = set(model.parameters())
            attn_params = []
            for name, param in model.named_parameters():
                if name == "encoder.attn_W_up" or name == "encoder.attn_W_down" or name == "encoder.layer_norm":
                    attn_params += list(param)
            attn_params = set(attn_params)
            non_attn_params = all_parameters - attn_params
            non_attn_params = list(non_attn_params)
            attn_params = list(attn_params)

            optim = AdamW([
                {'params': non_attn_params},
                {'params': attn_params, 'lr': model_args.attn_learning_rate},
            ], lr=training_args.learning_rate,)
            scheduler = get_linear_schedule_with_warmup(
                optim, num_warmup_steps=training_args.warmup_steps, num_training_steps=len(
                    train_dataset) * training_args.num_train_epochs // (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size)
            )
            return optim, scheduler

        _model = model
        if isinstance(wrapped_model, PTuningWrapper):
            model = model.underlying_model
        _lr = learning_rate
        no_decay = ['bias', 'LayerNorm.weight']
        router_learning_rate = float(router_lr)
        Az_learning_rate = 0.0001
        skill_params = [p for n, p in _model.named_parameters() if "skills_weight" in n]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in _model.named_parameters() if not "skills_weight" in n and p.requires_grad and not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay, "lr":_lr},
            {'params': [p for n, p in _model.named_parameters() if not "skills_weight" in n and p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr":_lr},
            {"params": skill_params, "lr": pl_learning_rate, "weight_decay": weight_decay}
            #{"params": model.mlp.parameters(), "lr": pl_learning_rate},
            #{"params": model.router, "lr": router_learning_rate},
            #{"params": model.A, "lr": Az_learning_rate},
            #{"params": model.z, "lr": Az_learning_rate},
        ]
        if opt_type == "adam":
            if (not isinstance(wrapped_model, PTuningWrapper) or
                len(wrapped_model.prompt_encoders) == 0):
                optimizer = AdamW(optimizer_grouped_parameters,eps=1e-8)
                scheduler = get_linear_schedule_with_warmup(optimizer,warm_up_steps,iterations)
            else:
                paras = []
                lrs = []
                for encoder in model.prompt_encoders:
                    if isinstance(encoder, MatPromptEncoder):
                        paras.append([encoder.router])
                        lrs.append(router_learning_rate)
                        paras.append([encoder.A])
                        lrs.append(Az_learning_rate)
                        paras.append([encoder.z])
                        lrs.append(Az_learning_rate)
                    else:
                        if encoder.router.requires_grad:
                            paras.append([encoder.router])
                            lrs.append(router_learning_rate)
                        para_list =[p for p in encoder.parameters() if p.requires_grad]
                        if para_list:
                            paras.append(para_list)
                            lrs.append(pl_learning_rate)
                for encoder in model.general_encoders:
                    if encoder.router.requires_grad:
                        paras.append([encoder.router])
                        lrs.append(router_learning_rate)
                    paras.append([p for p in encoder.parameters() if p.requires_grad])
                    lrs.append(pl_learning_rate)
                optimizer = Optim(paras, lrs)
                scheduler = Scheduler(optimizer)
            #if model.flat_encoder is not None:
            #    optimizer.add_param_group({'params': [p for p in model.flat_encoder.parameters() if p.requires_grad], "lr":pl_learning_rate})
            #optimizer.add_param_group({'params': model.A, "lr":Az_learning_rate})
            #optimizer.add_param_group({'params': model.z, "lr":Az_learning_rate})
            #optimizer.add_param_group({'params': model.router, "lr":router_learning_rate})
        elif opt_type == "ada_no_lr":
            optimizer = Adafactor(optimizer_grouped_parameters, 
                    scale_parameter=True, 
                    relative_step=True, warmup_init=True, lr=None)
            for encoder in model.prompt_encoders:
                optimizer.add_param_group({'params': [p for p in encoder.parameters() if p.requires_grad ], "lr":pl_learning_rate})
            if model.flat_encoder is not None:
                optimizer.add_param_group({'params': [p for p in model.flat_encoder.parameters() if p.requires_grad], "lr":pl_learning_rate})
            scheduler = AdafactorSchedule(optimizer)
        elif opt_type == "ada":
            mlog.info("Ada Factor")
            # replace AdamW with Adafactor
            optimizer = Adafactor(
                optimizer_grouped_parameters,
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
            for encoder in model.prompt_encoders:
                optimizer.add_param_group({'params': [p for p in encoder.parameters() if p.requires_grad ], "lr":pl_learning_rate})
            if model.flat_encoder is not None:
                optimizer.add_param_group({'params': [p for p in model.flat_encoder.parameters() if p.requires_grad], "lr":pl_learning_rate})
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

    mbp("start")
    optimizer, scheduler = get_optimizer(wrapped_model, learning_rate, opt_type) 
    if checkpoint:
        mlog.info("Restoring optimizer and scheduler")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        step = checkpoint['step']
        best_eval_step = checkpoint['best_eval_step']
        best_eval_loss = checkpoint['best_eval_loss']

    #%% build dataloader
    if data_name:
        train_ratio = 1
        dataset_path = os.path.join(data_path, data_name)
        train_dataloader, train_dataset, random_sampler = load_data2(dataset_path, "train", tokenizer, prompt_config, ratio=train_ratio, num=int(train_samples))
        for s in ["train","test","validation"]:
            load_data2(dataset_path, s, tokenizer, prompt_config, ratio=train_ratio, num=int(train_samples))
        train_records = int(train_samples)
    else:
        if "gpt" in model_id: 
            tokenizer.add_special_tokens(pad_token)
            tokenizer.add_special_tokens(sep_token)
            mlog.info("pad token id: %s", tokenizer.pad_token_id)
            data_collator = MyCollator(tokenizer, model, ds_type="train", model_type="gpt")
        else:
            data_collator = MyCollator(tokenizer, model, ds_type="train", prefix=prefix)

        train_dataset = myds["train"]#.map(tokenize)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
            batch_size=node_batch_size,shuffle=shuffle, num_workers=num_workers,
            collate_fn=data_collator,
        )
        if do_valid: 
            eval_dataset = myds["validation"]#.map(tokenize)
            eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                batch_size=node_batch_size,shuffle=shuffle,
                collate_fn=data_collator,
            )
    #torch.utils.data.DataLoader(myds['validation'],
    #    batch_size=node_batch_size,shuffle=shuffle,collate_fn=data_collator)
        train_records = train_dataset.num_records
    assert train_records != 0, "There is no data to train!!!!!!!!"
    for logger in [mlog, clog, vlog]:
        logger.info("Train records: %s", train_records)
    iterations = train_records//batch_size
    
    for logger in [mlog, ttlog]:
        logger.info("Iterations:"  + str(iterations))
    #st_embed = tf.saved_model.load("/home/pouramini/pret/sm")
    def consume(iterator, n):
        '''Advance the iterator n-steps ahead. If n is none, consume entirely.'''
        collections.deque(itertools.islice(iterator, n), maxlen=0)
    #11111111111
    # ffffffffffff
    #%% tttttt
    mlog.info(f"============== model : {type(wrapped_model)}\n")
    mlog.info(f"============== Example : {example}\n")
    mlog.info(f"============== Exp id: {exp_id}\n")
    mlog.info(f"============== Data Path: {data_path}\n")
    mlog.info(f"============== batch size: {batch_size} per node: {node_batch_size} | learning_rate: {learning_rate} | prompt_lr: {pl_learning_rate} \n")
    mlog.info(f"============== train samples: {train_samples} test_samples: {test_samples} | repeat: {repeat}  epochs: {epochs_num}\n")
    mlog.info(f"============== wrap: {wrap} | prefixed: {prefix} | frozen: {frozen} {freeze_parts}\n")
    mlog.info(f"============== rel_filter: {rel_filter} | method: {method} | model: {model_id} \n")
    epochs_num = int(epochs_num)
    cycle = int(cycle)
    wrap = True
    exp_info["enc_num"] = 0
    if isinstance(wrapped_model, PTuningWrapper): 
        exp_info["enc_num"] = len(wrapped_model.prompt_encoders) 
    exp_info["train_records"] = train_dataset.num_records
    exp_info["iterations"] = iterations 
    mbp("start")
    def train_loop(epochs_num, wrap, optimizer, scheduler):
        train_iter = iter(train_dataloader)
        global_step = 0
        max_acc = 0
        step = 0
        best_eval_loss = 100
        best_eval_step = 0
        best_step = 0
        is_freezed = frozen
        unfreeze_done = False
        freeze_done = False
        freeze_it = False
        unfreeze_it = False
        if epochs_num == 0 or (not wrap and frozen and modules_to_freeze is model):
            mlog.info("Skip training...")
        elif step <= iterations and (wrap or not frozen or modules_to_freeze is not model):
            mlog.info("Training... %s", save_path)
        for epoch in range(epochs_num):
            ttlog.info(f"============== epoch {epoch}\n")
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
                    if do_valid and cycle > 0 and (global_step % cycle == 0 and global_step > 0): #validation
                        _model = wrapped_model
                        with torch.no_grad():
                            mlog.info("Updating the model weights before evaluaton...")
                            wrapped_model.update_model_weight()
                        a1, a2, s1, eval_loss = evaluate1(tokenizer, eval_dataloader, _model, device, seed, mode="dev", save_path=save_path, task_ids=task_ids)
                        log_string =  "dev acc({}, {} st:{}): eval_loss:{}  | best_eval_loss:{}   best_step: {} ".format(a1, a2, s1, eval_loss, best_eval_loss, best_step) 
                        if eval_loss <= best_eval_loss: 
                            max_acc = a2
                            best_eval_loss = eval_loss
                            exp_info["max_acc"] = max_acc
                            exp_info["best_step"] = best_step = str(epoch) + "x" + str(step)
                            mlog.info(log_string)
                            if not no_save_best:
                                best_path = os.path.join(save_path, "best_model")
                                save_checkpoint(wrapped_model.underlying_model, 
                                        tokenizer, 
                                        optimizer, scheduler, step, 
                                        global_step, max_acc,
                                        best_path)

                            mbp(1)

                    try:
                        batch, no_model_batch = next(train_iter)
                    except StopIteration:
                        ttlog.info("Stop Iteration occured at %s", step)
                        train_iter = iter(train_dataloader)
                        batch, no_model_batch = next(train_iter)
                    batch = {k:v.to(device=device) for k,v in batch.items()}
                    freeze_it = no_model_batch["freeze"][0]
                    unfreeze_it = no_model_batch["unfreeze"][0]
                    if  unfreeze_it and not unfreeze_done:
                        mlog.info("unfreezing the model")
                        unfreeze_done = True 
                        unfreeze(modules_to_unfreeze)
                        last_lr = scheduler.get_last_lr()[0]
                        optimizer, scheduler = get_optimizer(wrapped_model, last_lr, opt_type)

                    if freeze_it and not freeze_done:
                        mlog.info("freezing the model")
                        freeze_done = True
                        freeze(modules_to_freeze)
                        last_lr = scheduler.get_last_lr()[0]
                        optimizer, scheduler = get_optimizer(wrapped_model, last_lr, opt_type)

                    ttlog.info("Wrap model zero grad")
                    wrapped_model.train()
                    wrapped_model.zero_grad()
                    optimizer.zero_grad()
                    _model = wrapped_model
                    out = forward_step(_model, batch, no_model_batch, task_ids=task_ids)
                    loss = out["loss"]
                    loss.backward()
                    if isinstance(wrapped_model, PTuningWrapper): 
                        for encoder in wrapped_model.prompt_encoders:
                            pass
                    optimizer.step()
                    scheduler.step()
                    step+=1
                    global_step+=1
                    bloss = loss.item()
                    tot_loss += bloss
                    mean_loss = tot_loss/(step-train_start)
                    sw.add_scalar('train/loss',bloss,global_step=step)
                    ttlog.info("{:<5}: {:6.2f} > {:6.2f}".format(step, bloss, mean_loss))
                    pbar.set_description(f'training ...[loss:{bloss:.2f} ({mean_loss:.2f}) best:{best_eval_step} {best_eval_loss:.2f}]')
                    pbar.update()
                    #del result
                    del loss
                except KeyboardInterrupt:
                    mlog.info("exiting while ...")
                    raise KeyboardInterrupt
                    break
        # end train while
        pbar.close()
        sw.close()
        #with torch.no_grad():
           # mlog.info("Updating the model weights before evaluaton...")
           # wrapped_model.update_model_weight()
    #% vvvv
    if loop == "custom": #not prefix:
        train_loop(epochs_num, wrap, optimizer, scheduler)
        mbp("start")
        # Initialize our Trainer
    elif loop == "atm":
        if model_args.attn_learning_rate is not None:
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset, 
                eval_dataset= eval_dataset, 
                evaluation_metrics=["rouge"],
                compute_metrics=compute_metrics if training_args.predict_with_generate else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                shared=model_args.shared_attn,
                optimizers=(optimizer, scheduler)
            )
        else:
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset= eval_dataset, 
                evaluation_metrics=["rouge"],
                compute_metrics=compute_metrics if training_args.predict_with_generate else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                shared=model_args.shared_attn)
        train_result = trainer.train()
    else:
        t_args = TrainingArguments(output_dir=save_path)
        t_args.per_device_train_batch_size=node_batch_size
        t_args.num_train_epochs=epochs_num
        t_args.save_strategy="steps"
        t_args.save_steps=10000 
        t_args.save_total_limit=1 

        #t_args.logging_steps=5
        #t_args.learning_rate=learning_rate
        t_args.report_to = ["wandb"] if wb else []
        t_args.do_predict=True
        t_args.gradient_accumulation_steps=accumulation_tiny_steps
        train_dataset = myds["train"]#.map(tokenize)
        if seed > 0:
            t_args.seed = seed
        #test_dataset = myds["test"]
        trainer = Seq2SeqTrainer(
            args=t_args,
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
        mbp("train")
        train_result = trainer.train()

    if model_args.save_prefix_only:
        save_prompts(wrapped_model, output_dir=training_args.output_dir, 
                attn_prefix_tuning=model_args.attn_prefix_tuning,
                shared_attn=model_args.shared_attn, 
                num_target=atm_config.num_target, task_name=data_args.task_name)
    if False: #not no_save_model:
        model.eval()
        save_checkpoint(wrapped_model.underlying_model, tokenizer, 
                optimizer, scheduler, step, 
                best_eval_step, best_eval_loss,
                save_path)
    else:
        mlog.info("No save model is on!!")
    # vvvv
    if False: #do_valid and not no_save_best:
        mlog.info("loading best model")
        best_path = os.path.join(save_path, "best_model")
        model, tokenizer, _, atm_config = load_model(model_id, best_path) 
        if no_save_model:
            shutil.rmtree(best_path)
        model.to(device)
    model.eval()
    #vvvvvv
    if data_name:
        test_ratio = 1
        _set = "test"
        dataset_path = os.path.join(data_path, data_name)
        test_dataloader, test_dataset, random_sampler = load_data2(dataset_path, "test", tokenizer, prompt_config, ratio=test_ratio, num=int(test_samples))
        val_records = int(test_samples)
        evaluate1(tokenizer, test_dataloader, model, device, prompt_config, mode="test", save_path="", task_ids=task_ids)
        evaluate(test_dataset, test_dataloader, save_path, exp_info, val_records, gen_param, scorers = scorers, batch_size=gen_bs, model=wrapped_model, tokenizer=tokenizer, set_name=_set, break_point=break_point, seed=seed, task_ids=task_ids)  
    elif test_set:
        eval_test(model, tokenizer)
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
        dataPath = "atomic" 
    else:
        logPath = os.path.join(home, "logs")
        resPath = os.path.join(home, "results") 
        pretPath = os.path.join(home, "pret") 
        dataPath = os.path.join(home, "atomic") 

    base_dir = home
    if "_" in experiment:
        raise ValueError("Experiment name shouldn't have underscore in it, use dash")
    conf = "base_confs/test.json" 
    save_path = os.path.join(base_dir, "mt5-comet/comet/train/")
    if not is_colab:
        conf_path = os.path.join(save_path,"confs",experiment)
        assert False, conf_path
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
    #args["cycle"] = 0
    args["no_save_model"] = not save_model
    args["load_path"] = pretPath
    args["train_path"] = "atomic/train.tsv"
    save_path = os.path.join(logPath, experiment)
    args["save_path"] = save_path

    args["cpu"] = False 
    args["config"] = "" 
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
