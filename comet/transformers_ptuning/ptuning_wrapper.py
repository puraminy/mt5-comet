import wandb
import re
from pathlib import Path
import transformers
import numpy as np
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import logging
import os
import math
from os.path import expanduser
import comet.train.mylogs as logs
from comet.train.mylogs import tinfo, getFname, tlog
from comet.transformers_ptuning.encoders import * 

from comet.train.mylogs import mbp 
from transformers.optimization import Adafactor, AdamW
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


def _isin(tensor:torch.Tensor,values:torch.Tensor):
    return (tensor[..., None] == values).any(-1)
home = expanduser("~")
wlog = logging.getLogger("comet.wrapper")
emblog = logging.getLogger("comet.embedding")
consoleHandler = logging.StreamHandler()
#wlog.addHandler(consoleHandler)
#emblog.addHandler(consoleHandler)
FORMAT = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)10s() ] %(message)s")

def winfo(text, *args, **kwargs):
    wlog.info(text, *args)
    #print((text, *args))

def embinfo(text, *args, **kwargs):
    emblog.info(text, *args)
    #print((text, *args))

class PTuningWrapper(torch.nn.Module):
    def __init__(self,model,prompt_encoders, general_encoders,decoder_prompt_encoder=None,
        prompt_token_fn=None, prompt_token_id=None, prompt_token_ids=None,
        replacing_token_id=0, do_log=True, merge_encoder=None, 
        flat_encoder = None, args={}):
        """
        PTuningWrapper for Huggingface transformer models (Encoder Models).
        It will replace the prompt token embeddings with ones from prompt encoder.
        Therefore the origin embedding layer can be freezed.

        Parameters:
            model: 
                The huggingface transformer model to be wrapped.
            prompt_encoder: 
                A model that returns corresponding embeddings for prompt ids.
            decoder_prompt_encoder:
                Similar to prompt_encoder, but used for decoder part of the 
                model if the model is encoder-decoder (e.g. T5 and BART).
            prompt_token_fn:
                A function that receives the input_ids and returns a boolean
                tensor to tell which ones are prompt tokens. This parameter 
                conflicts with prompt_token_id and prompt_token_ids.
            prompt_token_id:
                If there is only one possible prompt token id, use this
                parameter.
            prompt_token_ids:
                If there is a list of prompt token ids, use this paramter.
            replacing_token_id:
                During processing, all prompt token will be replaced with this
                one so that the input_ids can be passed into the model 
                embedding layer of the transformer model.
        """

        wlog.handlers.clear()
        emblog.handlers.clear()
        #wHandler = logging.FileHandler(getFname(exp + "_wrapper"), mode='w')
        #wHandler.setFormatter(FORMAT)
        #wlog.addHandler(wHandler)
        #eHandler = logging.FileHandler(getFname(exp + "_embedding"), mode='w')
        #eHandler.setFormatter(FORMAT)
        #emblog.addHandler(eHandler)

        embinfo("Embedding log")
        winfo("Wrapper log")

        wlog.setLevel(logging.INFO)
        emblog.setLevel(logging.INFO)


        super().__init__()
        mbp("")
        cpu = args["cpu"]
        self.device = 'cuda' if not cpu and torch.cuda.is_available() else 'cpu'
        self.flat_encoder = flat_encoder
        self.merge_encoder = merge_encoder
        #assert flat_encoder == None, "merege_encoder was set"
        if flat_encoder:
            self.flat_prompt_ids = flat_encoder.prompt_ids
        self.testing = False
        if not do_log or not "ahmad" in home:
            wlog.disabled = False
            self.testing = False
        self.ll = logging.INFO
        if self.testing:
            winfo("%%%%%%%%%%%%%%%%%%% testing is ON %%%%%%%%%%%%%%%%%%")
        winfo("%%%%%%%%%%%%%%%%%%%%%%%% Wrapper log %%%%%%%%%%%%%%%%%%")
        tlog.info("%%%%%%%%%%%%%%%%%%%%%%%% Time log %%%%%%%%%%%%%%%%%%")
        self.underlying_model = model
        self.model_embeddings = model.get_input_embeddings()
        winfo("self.model embedding:{}".format(self.model_embeddings))
        model_embeddings_size = model.get_input_embeddings().num_embeddings
        winfo("model embedding_size:{}".format(model_embeddings_size))
        self.prompt_encoders = torch.nn.ModuleList(prompt_encoders)
        self.general_encoders = torch.nn.ModuleList(general_encoders)
        winfo("num of encoders %s:", len(self.prompt_encoders))
        self.config = model.config
        self.embedding_dim = model.config.hidden_size
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.decoder_prompt_encoder = decoder_prompt_encoder
        self.decoder_prompt_flag = False
        self.encoder_prompt_flag = False
        self.replacing_token_id = replacing_token_id
        winfo("REP id:{}".format(replacing_token_id))
        self.model_decoder_embeddings = model.decoder.embed_tokens
        wlog.debug("DECODER model embd:{}".format(self.model_decoder_embeddings))
        if prompt_token_fn is not None:
            assert prompt_token_id is None and prompt_token_ids is None, \
                "Use one of prompt_token_fn, prompt_token_id, prompt_token_ids"
            self.prompt_token_fn = prompt_token_fn
        elif prompt_token_ids is not None:
            assert prompt_token_id is None, \
                "Use one of prompt_token_fn, prompt_token_id, prompt_token_ids"
            self.prompt_token_ids = torch.nn.parameter.Parameter(
                torch.tensor(prompt_token_ids,device=model.device))
            self.prompt_token_fn = lambda t:_isin(t,self.prompt_token_ids)
        elif prompt_token_id is not None:
            self.prompt_token_id = prompt_token_id
            self.prompt_token_fn = lambda t:(t==prompt_token_id)
        else:
            wlog.debug("ITTTTTTTTTT doesn't come here")
            # self.model_embeddings_size = self.model.get_input_embeddings()\
            #     .num_embeddings
            self.model_embeddings_size = self.underlying_model.config.vocab_size
            self.prompt_token_fn = lambda t:(t>=self.model_embeddings_size)
            

    def has_encoders():
        return True

    def add_prompt_encoder(self, encoder):
        self.prompt_encoders.append(encoder)

    def encoder_forward(self, encoder, input_ids, inputs_embeds, tids):
        #encoder = self.prompt_encoders[0]
        device=inputs_embeds.device
        prompt_token_fn = encoder.get_prompt_token_fn()
        encoder_masks = prompt_token_fn(input_ids)
        if encoder_masks.any():
            #find input ids for prompt tokens
            prompt_input_ids = input_ids[encoder_masks]
            # call forwards on prompt encoder whose outputs are prompt embeddings
            prompt_embeds = encoder(prompt_input_ids, tids).to(device)
            inputs_embeds[encoder_masks]=prompt_embeds
            return prompt_embeds
        return None

    def generate(self, input_ids, *args, **kwargs):
        if logs.args("stype") == "atm":
            task_ids = kwargs.setdefault("task", None)
        else:
            task_ids = kwargs.pop("task", None)
        tinfo("gen task ids ggggggggggg: %s", task_ids)
        device = input_ids.device
        #task_ids = torch.tensor([0])
        if task_ids != None:
            task_ids = task_ids.long()
            task_ids.to(device)
        #inform_layers(self.underlying_model, self.adapter_class, task_ids)
        self.update_model_weight(task_ids)
        return self.underlying_model.generate(input_ids, *args, **kwargs)

    def forward(self,input_ids, tids=None, **kwargs):
        ll = self.ll # log level
        # find masks based on the range of prompt ids (offset_id < X < offset_id + prompt_length)
        #Because this wrapper only deals with a single prompt, the length should be the same, you can use masked_select to reshape 
        tids = kwargs.pop("task", None)
        tids = tids.long()
        append_prefix = (hasattr(self.underlying_model, "prefix_tuning") 
                         and self.underlying_model.prefix_tuning and 
                         not self.underlying_model.attn_prefix_tuning)
        if append_prefix:
            return self.underlying_model(input_ids=input_ids, **kwargs)

        prompt_masks = self.prompt_token_fn(input_ids)
        if prompt_masks.any():
            self.encoder_prompt_flag = True
            input_ids_clone = input_ids.clone()
            if self.replacing_token_id is not None:
                # replace prompt ids in input_ids with replacing token
                input_ids_clone[prompt_masks]=self.replacing_token_id
            # find the model embeddings of input ids except for prompt tokens
            inputs_embeds = self.model_embeddings(input_ids_clone)
            device=inputs_embeds.device
            all_prompts_input_ids = input_ids[prompt_masks]
            winfo("All prompts input ids: %s", all_prompts_input_ids)
            winfo("Len All prompts input ids: %s", len(all_prompts_input_ids))
            if self.merge_encoder:
                #flat_output = self.flat_encoder(all_prompts_input_ids,tids).to(device)
                merge_output = self.encoder_forward(self.merge_encoder, input_ids, inputs_embeds, tids)
            elif self.flat_encoder:
                #flat_output = self.flat_encoder(all_prompts_input_ids,tids).to(device)
                flat_output = self.encoder_forward(self.flat_encoder, input_ids, inputs_embeds, tids)
            else:
                embeds_list = []
                flat_dict = {}
                for encoder in self.prompt_encoders:
                    prompt_embeds = self.encoder_forward(encoder, input_ids, inputs_embeds, tids)
                    #inputs_embeds[encoder_masks]=prompt_embeds
                    if self.testing:
                        for _id,_embed in zip(prompt_input_ids.numpy(),prompt_embeds):
                            _key = encoder.name + "_" + str(_id)
                            if not _key in flat_dict:
                                flat_dict[_key] = [_embed]
                            
                        embeds_list.append(prompt_embeds)
                    # replace prompt_embeddings calculated by prompt encoder in input embeddings
                if self.testing:
                    for key,val in flat_dict.items():
                        winfo("Merge dict item %s, len: %s",key, len(val))
                        if val:
                            flat_dict[key] = self.mlp(torch.stack(val))
                    winfo("embeds list: %s", len(embeds_list))
                    res_embeds = self.mlp(torch.cat(embeds_list))
                    winfo("REES embeds: %s", res_embeds)
                    winfo("REES embeds size: %s", res_embeds.size())
                    winfo("All prompts input ids: %s", all_prompts_input_ids)
                    winfo("PROMPT MASKS: %s", prompt_masks)
                    winfo("Merge dict: %s", flat_dict)
        else:
            inputs_embeds = self.model_embeddings(input_ids)

        decoder_input_ids = kwargs.pop("decoder_input_ids", None) 
        if decoder_input_ids is not None:
            mbp("")
            prompt_masks = self.prompt_token_fn(decoder_input_ids)
            if prompt_masks.any():
                self.decoder_prompt_flag = True
                labels = kwargs.pop("labels", None) 
                winfo("promp masks:{}".format(prompt_masks))
                decoder_input_ids_clone = decoder_input_ids.clone()
                winfo("inpu ids :{}".format(decoder_input_ids))
                if self.replacing_token_id is not None:
                    # replace prompt ids in input_ids with replacing token
                    decoder_input_ids_clone[prompt_masks] = \
                        self.replacing_token_id
                # find the model embeddings of input ids except for prompt tokens
                decoder_inputs_embeds = self.model_decoder_embeddings(
                    decoder_input_ids_clone
                )
                all_prompts_input_ids = decoder_input_ids[prompt_masks]
                for encoder in self.prompt_encoders:
                    #encoder = self.prompt_encoders[0]
                    winfo("********** offset: %s, length: %s", encoder.id_offset, encoder.length)
                    prompt_token_fn = encoder.get_prompt_token_fn()
                    encoder_masks = prompt_token_fn(decoder_input_ids)
                    winfo("Encoder masks: %s", encoder_masks)
                    if encoder_masks.any():
                        #find input ids for prompt tokens
                        prompt_input_ids = decoder_input_ids[encoder_masks]
                        winfo("Prompt Input ids: %s", prompt_input_ids)
                        winfo("Len Prompt Input ids: %s", len(prompt_input_ids))
                        # call forwards on prompt encoder whose outputs are prompt embeddings
                        device=decoder_input_ids.device
                        decoder_prompt_embeds = encoder(prompt_input_ids, tids).to(device)

                        decoder_inputs_embeds[encoder_masks] = \
                            decoder_prompt_embeds
                    
                        decoder_labels_masks = self.prompt_token_fn(labels)
                        labels[decoder_labels_masks] = -100
                        winfo(labels)
                        break
                else:
                    decoder_inputs_embeds = self.model_decoder_embeddings(
                        decoder_input_ids
                    )
                return self.underlying_model(inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, labels=labels, **kwargs)
            else: 
                self.ll = logging.DEBUG
                return self.underlying_model(inputs_embeds=inputs_embeds, **kwargs)
        else:
            self.ll = logging.DEBUG
            return self.underlying_model(inputs_embeds=inputs_embeds,**kwargs)
    def update_model_weight(self, task_ids = None):
        self.cur_embeddings = self.underlying_model.get_input_embeddings()
        if self.merge_encoder:
            self.merge_encoder.dump_embeddings_into(self.cur_embeddings.weight, task_ids)
        elif self.flat_encoder:
            self.flat_encoder.dump_embeddings_into(self.cur_embeddings.weight, task_ids)
        elif self.encoder_prompt_flag:
            for encoder in self.prompt_encoders:
                # fill the current embeddings with weights of encoder
                encoder.dump_embeddings_into(self.cur_embeddings.weight, task_ids)
                #self.prompt_encoder.dump_embeddings_into(self.model_embeddings.weight)
        if self.decoder_prompt_encoder in self.prompt_encoders:
            winfo(f"Encoder and Decoder are the same")
            pass
        if self.decoder_prompt_flag:
            for encoder in self.prompt_encoders:
                winfo(f"the wrapper has prompt encoder for decoder part")
                # fill the current embeddings with weights of encoder
                encoder.dump_embeddings_into(
                                       self.model_decoder_embeddings.weight, 
                                       task_ids)
                #self.prompt_encoder.dump_embeddings_into(self.model_embeddings.weight)
        if self.decoder_prompt_encoder:
            self.decoder_prompt_encoder.dump_embeddings_into(
                self.model_decoder_embeddings.weight, task_ids)

