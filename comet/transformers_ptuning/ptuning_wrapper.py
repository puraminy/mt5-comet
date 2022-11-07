#暂时没考虑encoder和decoder的tokenizer不同的情况，以后可以给decoder全套的prompt_token_fn

import re
from pathlib import Path
import transformers
import numpy as np
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
def _isin(tensor:torch.Tensor,values:torch.Tensor):
    return (tensor[..., None] == values).any(-1)
import logging
import os
import math
from os.path import expanduser
from comet.train.mylogs import mbp
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

home = expanduser("~")
wlog = logging.getLogger("comet.wrapper")
emblog = logging.getLogger("comet.embedding")
consoleHandler = logging.StreamHandler()
wlog.addHandler(consoleHandler)
emblog.addHandler(consoleHandler)
tlog = logging.getLogger("comet.time")
FORMAT = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)10s() ] %(message)s")

def winfo(text, *arg, **kwargs):
    pass
    #print((text, *arg))

def embinfo(text, *arg, **kwargs):
    pass
    #print((text, *arg))

def getFname(name):
    if "ahmad" in home or "pouramini" in home:
        logFilename = os.path.join(home, f"logs/{name}.log")
    else:
        logFilename = f"{name}.log"
    return logFilename
wHandler = logging.FileHandler(getFname("wrapper"), mode='w')
wHandler.setFormatter(FORMAT)
wlog.addHandler(wHandler)
eHandler = logging.FileHandler(getFname("embedding"), mode='w')
eHandler.setFormatter(FORMAT)
emblog.addHandler(eHandler)
tHandler = logging.FileHandler(getFname("time"), mode='w')
tHandler.setFormatter(FORMAT)
tlog.addHandler(tHandler)
embinfo("Embedding log")
winfo("Wrapper log")
wlog.setLevel(logging.INFO)
emblog.setLevel(logging.INFO)
tlog.setLevel(logging.INFO)

class PTuningWrapper(torch.nn.Module):
    def __init__(self,model,prompt_encoders,decoder_prompt_encoder=None,
        prompt_token_fn=None, prompt_token_id=None, prompt_token_ids=None,
        replacing_token_id=0, do_log=True, 
        merge_prompts = "", shared_embs = False, prefix_config=None):
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
        super().__init__()
        mbp("")
        self.testing = False
        if merge_prompts == "none": merge_prompts = ""
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
        winfo("num of encoders %s:", len(self.prompt_encoders))
        self.config = model.config
        self.embedding_dim = model.config.hidden_size
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.merge_prompt_ids = []
        sum_len = 0
        for encoder in self.prompt_encoders:
            _ids = encoder.prompt_ids
            sum_len += len(_ids)
            for _id in _ids:
                if not _id in self.merge_prompt_ids:
                    self.merge_prompt_ids.append(_id)

            _offset = min(_ids)
            winfo("** existing encoder ids for %s: %s", encoder.name, _ids)
            rel_ids_tensor = torch.LongTensor(_ids)
            embs = self.model_embeddings
            rel_embs = embs(rel_ids_tensor)
            with torch.no_grad():
               for i, e in enumerate(rel_embs):
                   encoder.embedding.weight[i] = e #.detach()

        if self.prompt_encoders:
            self.merge_offset = min(self.merge_prompt_ids)
            winfo("Offset: %s,", self.merge_offset)
        winfo("Merge ids: %s,", self.merge_prompt_ids)
        winfo("Merge ids len: %s,", len(self.merge_prompt_ids))
        winfo("Sum len: %s,", sum_len)

        self.merge_encoder = None 
        self.merge_embedding = None
        n_prompt_tokens = len(self.merge_prompt_ids)
        mbp("")
        if merge_prompts:
            _enc_type = merge_prompts.split("@")
            num_layers = 1
            if len(_enc_type) > 1:
                num_layers = int(_enc_type[1])
            hidden_size = -1
            if len(_enc_type) > 2:
                hidden_size = int(_enc_type[2])
            if merge_prompts.startswith("mlp"):
                self.merge_encoder = MLPPromptEncoder("wrap_all", len(self.merge_prompt_ids), self.embedding_dim, self.merge_offset, prompt_ids=self.merge_prompt_ids, num_layers=num_layers, hidden_size=hidden_size)
            elif merge_prompts.startswith("lstm"):
                self.merge_encoder = LSTMEmbeddingPromptEncoder("wrap_all", len(self.merge_prompt_ids), self.embedding_dim, self.merge_offset, prompt_ids=self.merge_prompt_ids, num_layers=num_layers, hidden_size=hidden_size)
            assert self.merge_encoder != None, "merge encoder for " + merge_prompts + " is none"
        else:
            if shared_embs:
                self.merge_embedding = torch.nn.Embedding(len(self.merge_prompt_ids), self.embedding_dim)
                for encoder in self.prompt_encoders:
                    encoder.embedding = self.merge_embedding
                    encoder.id_offset= self.merge_offset
                    encoder.length= len(self.merge_prompt_ids)


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
            
        if prefix_config is None:
            prefix_config = {
                  "temperature": 0.3,
                  "n_tasks": len(prompt_encoders),
                  "n_prompts": len(self.merge_prompt_ids),
                  "n_prompt_tokens": len(self.merge_prompt_ids),
                  "intrinsic_dim": 300,
            }
        self.prefix_config = prefix_config
        if prefix_config is not None:
            self.temperature = self.prefix_config['temperature']
            self.task_id = 0

            self.router = nn.Parameter(data=torch.empty((
                prefix_config['n_tasks'],
                prefix_config['n_prompts']
            )).uniform_(-1e-3, 1e-3))

            self.z = nn.Parameter(data=torch.empty((
                prefix_config['n_prompts'],
                prefix_config['intrinsic_dim']
            )).uniform_(-1e-3, 1e-3))

            bound = 1 / math.sqrt(prefix_config['n_prompt_tokens'] * model.config.hidden_size)
            self.A = nn.Parameter(data=torch.empty((
                prefix_config['intrinsic_dim'],
                prefix_config['n_prompt_tokens'] * model.config.hidden_size
            )).uniform_(-bound, bound))


    def add_prompt_encoder(self, encoder):
        self.prompt_encoders.append(encoder)

    def forward(self,input_ids, pids=None, **kwargs):
        ll = self.ll # log level
        winfo("wrapper forward was called")
        winfo("Prompt ids:{}".format(pids))
        # find masks based on the range of prompt ids (offset_id < X < offset_id + prompt_length)
        #Because this wrapper only deals with a single prompt, the length should be the same, you can use masked_select to reshape 
        tids = kwargs.pop("task", None)
        tids = tids.long()
        prompt_masks = self.prompt_token_fn(input_ids)
        if prompt_masks.any():
            self.encoder_prompt_flag = True
            winfo("promp masks:{}".format(prompt_masks))
            input_ids_clone = input_ids.clone()
            winfo("inpu ids :{}".format(input_ids))
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
                merge_output = self.merge_encoder(all_prompts_input_ids,pids).to(device)
                if True: #self.prefix_config is None:
                    inputs_embeds[prompt_masks]=merge_output
                else:
                    router = torch.index_select(self.router, 0, tids)
                    if self.training:
                        router = RelaxedBernoulli(temperature=self.temperature, logits=router).rsample()  # layer * n_prompts
                    else:
                        router = torch.sigmoid(router)  # layer * n_prompts
                    router = (router / (router.sum(dim=-1, keepdim=True) + 1e-12))  # layer * 1 * n_prompts
                    z = torch.mm(self.z, self.A) if not hasattr(self, 'prompt') else self.prompt
                    #prompt_embeds = torch.matmul(router.unsqueeze(0), z).view(-1, self.config.hidden_size)
                    prompt_embeds = torch.matmul(router.view(1, -1), merge_output).view(-1, self.config.hidden_size)


                    inputs_embeds[prompt_masks]=merge_output
                #prompt_padding = torch.zeros(size=(hidden_shape[0], hidden_shape[1] - self.prefix_config['n_prompt_tokens'] - 1, self.config.hidden_size), device=device)

                #extended_prompt_embedding = torch.cat([prompt_embedding, prompt_padding], dim=1)
                #pre_padding = torch.zeros(size=(hidden_shape[0], 1, self.config.hidden_size), device=device)
                #extended_prompt_embedding = torch.cat([pre_padding, extended_prompt_embedding], dim=1)  # for <CLS>
                # extended_prompt_embedding = extended_prompt_embedding.repeat(input_shape[0], 1, 1)
                #hidden_states = hidden_states + extended_prompt_embedding
            else:
                embeds_list = []
                merge_dict = {}
                for encoder in self.prompt_encoders:
                    #encoder = self.prompt_encoders[0]
                    winfo("********** offset: %s, length: %s", encoder.id_offset, encoder.length)
                    prompt_token_fn = encoder.get_prompt_token_fn()
                    encoder_masks = prompt_token_fn(input_ids)
                    winfo("Encoder masks: %s", encoder_masks)
                    if encoder_masks.any():
                        #find input ids for prompt tokens
                        prompt_input_ids = input_ids[encoder_masks]
                        winfo("Prompt Input ids: %s", prompt_input_ids)
                        winfo("Len Prompt Input ids: %s", len(prompt_input_ids))
                        # call forwards on prompt encoder whose outputs are prompt embeddings
                        prompt_embeds = encoder(prompt_input_ids,\
                            pids).to(device)
                        if self.testing:
                            for _id,_embed in zip(prompt_input_ids.numpy(),prompt_embeds):
                                _key = encoder.name + "_" + str(_id)
                                if not _key in merge_dict:
                                    merge_dict[_key] = [_embed]
                                
                            embeds_list.append(prompt_embeds)
                        # replace prompt_embeddings calculated by prompt encoder in input embeddings
                        winfo("Prompt Embeds size: %s", prompt_embeds.size())
                        winfo("Encoder mask: %s", encoder_masks.size())
                        inputs_embeds[encoder_masks]=prompt_embeds
                if self.testing:
                    for key,val in merge_dict.items():
                        winfo("Merge dict item %s, len: %s",key, len(val))
                        if val:
                            merge_dict[key] = self.mlp(torch.stack(val))
                    winfo("embeds list: %s", len(embeds_list))
                    res_embeds = self.mlp(torch.cat(embeds_list))
                    winfo("REES embeds: %s", res_embeds)
                    winfo("REES embeds size: %s", res_embeds.size())
                    winfo("All prompts input ids: %s", all_prompts_input_ids)
                    winfo("PROMPT MASKS: %s", prompt_masks)
                    winfo("Merge dict: %s", merge_dict)
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
                        decoder_prompt_embeds = encoder(prompt_input_ids,\
                            pids).to(device)

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
    def update_model_weight(self):
        winfo(f"Updating model weights")
        self.cur_embeddings = self.underlying_model.get_input_embeddings()
        if self.merge_encoder:
            self.merge_encoder.dump_embeddings_into(self.cur_embeddings.weight)
        elif self.encoder_prompt_flag:
            for encoder in self.prompt_encoders:
                winfo(f"the wrapper has prompt encoder")
                # fill the current embeddings with weights of encoder
                encoder.dump_embeddings_into(self.cur_embeddings.weight)
                #self.prompt_encoder.dump_embeddings_into(self.model_embeddings.weight)
        if self.decoder_prompt_encoder in self.prompt_encoders:
            winfo(f"Encoder and Decoder are the same")
            pass
        if self.decoder_prompt_flag:
            for encoder in self.prompt_encoders:
                winfo(f"the wrapper has prompt encoder")
                # fill the current embeddings with weights of encoder
                encoder.dump_embeddings_into(
                                       self.model_decoder_embeddings.weight)
                #self.prompt_encoder.dump_embeddings_into(self.model_embeddings.weight)
        if self.decoder_prompt_encoder:
            self.decoder_prompt_encoder.dump_embeddings_into(
                self.model_decoder_embeddings.weight)


class PromptEncoder(torch.nn.Module):
    def __init__(self,name, length,embedding_dim,id_offset, init_embs, prompt_ids,**kwargs) -> None:
        super().__init__()
        self.length = length
        self.name = name
        self.counter = 0
        self.prompt_ids = prompt_ids
        self.input_ids = torch.nn.parameter.Parameter(torch.tensor(prompt_ids),
             requires_grad=False)
        embinfo("=========================== %s ===================", name)
        embinfo("prompt ids: %s", prompt_ids)
        self.embedding_dim = embedding_dim
        self.id_offset = id_offset
        self.embedding = torch.nn.Embedding(length,embedding_dim)
        #self.embedding.weight.requires_grad = False
        if init_embs:
            with torch.no_grad():
                for _id,emb in init_embs.items():
                    if _id < len(self.embedding.weight):
                        self.embedding.weight[_id] = emb
                        embinfo("%s : %s", _id, emb)

    def isin(self, ar1, ar2):
        return (ar1[..., None] == ar2).any(-1)
    def get_prompt_token_fn(self):
        if self.input_ids is not None:
            return lambda x: self.isin(x, self.input_ids)
        else:
            return lambda x: (x>=self.id_offset)&(x<self.id_offset+self.length)
    def dump_embeddings_into(self,weight):
        raise NotImplementedError


class EmbeddingPromptEncoder(PromptEncoder):
    def __init__(self,name,length,embedding_dim,id_offset,init_embs=None, prompt_ids=[]) -> None:
        super().__init__(name, length,embedding_dim,id_offset, init_embs, prompt_ids)
    
    def forward(self,prompt_token_ids,pids=None):
        embinfo("=========================== Forward ===================")
        embinfo("=========================== %s ===================", self.name)
        embinfo("Before prompt token ids: %s", prompt_token_ids)
        #embinfo("id offset: %s", self.id_offset)
        #embinfo("id length: %s", self.length)
        if self.id_offset > 0:
            prompt_token_ids = prompt_token_ids - self.id_offset
        else:
            prompt_token_ids = (prompt_token_ids.view(-1,1) == self.input_ids).int().argmax(dim=1)
        embinfo("self input ids: %s", self.input_ids)
        embinfo("After prompt token ids: %s", prompt_token_ids)
        embinfo(self.embedding.weight)
        ret_embs = self.embedding(prompt_token_ids)
        embinfo("ret embs %s", ret_embs)
        embinfo("=========================== Forward end ===================")
        return ret_embs

    def dump_embeddings_into(self, weight):
        winfo("Dump embeddings")
        embinfo("=========================== %s ===================", self.name)
        embinfo("input weights: %s", weight)
        detached_embeddings = self.embedding.weight.detach()
        embinfo("Dump embeddings: %s", detached_embeddings)
        embinfo("on this ids: %s", self.prompt_ids)
        weight[self.prompt_ids,:]=detached_embeddings

class MLPPromptEncoder(PromptEncoder):
    def __init__(self,name,length,embedding_dim,id_offset,init_embs=None, prompt_ids=[], num_layers=1, hidden_size=-1) -> None:
        super().__init__(name, length,embedding_dim,id_offset, init_embs, prompt_ids)
        hsize = hidden_size if hidden_size > 1 else embedding_dim
        if num_layers == 2:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, hsize),
                torch.nn.ReLU(),
                torch.nn.Linear(hsize, hsize),
                torch.nn.ReLU(),
                torch.nn.Linear(hsize, embedding_dim)
            )
        else:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, hsize),
                torch.nn.ReLU(),
                torch.nn.Linear(hsize, embedding_dim)
            )
    
    def forward(self,prompt_token_ids,pids=None):
        embinfo("=========================== Forward ===================")
        embinfo("=========================== %s ===================", self.name)
        embinfo("Before prompt token ids: %s", prompt_token_ids)
        #embinfo("id offset: %s", self.id_offset)
        #embinfo("id length: %s", self.length)
        if self.id_offset > 0:
            prompt_token_ids = prompt_token_ids - self.id_offset
        else:
            prompt_token_ids = (prompt_token_ids.view(-1,1) == self.input_ids).int().argmax(dim=1)
        embinfo("self input ids: %s", self.input_ids)
        embinfo("After prompt token ids: %s", prompt_token_ids)
        embinfo(self.embedding.weight)
        embs = self.embedding(prompt_token_ids)
        ret_embs = self.mlp(embs)
        embinfo("ret embs %s", ret_embs)
        embinfo("=========================== Forward end ===================")
        return ret_embs

    def dump_embeddings_into(self, weight):
        winfo("Dump embeddings")
        embinfo("=========================== %s ===================", self.name)
        with torch.no_grad():
            embs = self.forward(self.input_ids)
        embinfo("input weights: %s", weight)
        detached_embeddings = embs.detach()
        embinfo("Dump embeddings: %s", detached_embeddings)
        embinfo("on this ids: %s", self.prompt_ids)
        weight[self.prompt_ids,:]=detached_embeddings

class LSTMEmbeddingPromptEncoder(PromptEncoder):
    def __init__(self,name, length,embedding_dim,id_offset, init_embs=None, prompt_ids=[], num_layers=1, hidden_size=-1) -> None:
        super().__init__(name, length,embedding_dim,id_offset, init_embs, prompt_ids)
        hsize = hidden_size if hidden_size > 1 else embedding_dim
        self.net_inps = torch.nn.parameter.Parameter(torch.arange(length),
            requires_grad=False)
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim // 2, #my code
            num_layers=2,
            dropout=0,
            bidirectional=True,
            batch_first=True
        )
        if num_layers == 2:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, embedding_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(embedding_dim, hsize),
                torch.nn.ReLU(),
                torch.nn.Linear(hsize, embedding_dim)
            )
        else:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, hsize),
                torch.nn.ReLU(),
                torch.nn.Linear(hsize, embedding_dim)
            )

 #### llllllf
    def forward(self,prompt_token_ids,pids=None):
        embinfo("=========================== Forward begin ===================")
        embinfo("=========================== %s ===================", self.name)
        embinfo("before prompt token ids:{}".format(prompt_token_ids))
        embinfo("self input ids: %s", self.input_ids)
        embinfo("NETTTTT inps:{}".format(self.net_inps))
        # find zero based ids 
        net_inputs = self.net_inps
        if self.id_offset > 0:
            embinfo("------------------- case id offset > 0 ----------------")
            net_inputs = self.input_ids - self.id_offset
            #index_list = [((net_inputs == x).nonzero(as_tuple=True)[0]) for x in prompt_token_ids_2]
        index_list = (prompt_token_ids.view(-1,1) == self.input_ids).int().argmax(dim=1)
        embinfo("after prompt token ids:  %s", prompt_token_ids)
        embinfo("after net inputs:  %s", net_inputs)
        embinfo("index list:  %s", index_list)
        # create embedding vectors for input ids
        embeds = self.embedding(net_inputs)
        # do forward calculations
        if self.name == "xIntent":
            tlog.info("LSTM: %s", self.name)
            tlog.info("after net inputs:  %s", net_inputs)
            tlog.info("after prompt token ids:  %s", prompt_token_ids)
            tlog.info("lstm embeds: %s",embeds)

        x = self.lstm(embeds.unsqueeze(0))
        #embinfo("XXXXXXXXXXXXXXXXX: %s",x)
        #embinfo("XXXXXXXXXXXXXXXXX[0]: %s",x[0])
        #embinfo("XXXXXXXXXXXXXXXXX size: %s",x[0].size())
        embinfo("lstm embeds: %s",embeds)

        running_weight = self.mlp(x[0]).squeeze(0)
        if self.counter < 5:
            embinfo("--------------------")
            embinfo("running weights: %s",running_weight)
            embinfo("running weights size: %s",running_weight.size())
            self.counter += 1

        # return weights for prompt_token_ids 
        ret_embeds = F.embedding(index_list,running_weight)
        embinfo("ret embeds size %s", ret_embeds.size())
        embinfo("ret embeds %s", ret_embeds)
        embinfo("=========================== Forward end ===================")
        return ret_embeds
    def dump_embeddings_into(self, weight):
        # get embedding weights as the output of forward pass
        embinfo("%%%%%%%%%%%%%%%%%%%%%%%%%% dump embeddings start %%%%%%%%%%%%%%%%")
        embinfo("=========================== %s ===================", self.name)
        embinfo("Dump embeddings: %s", weight)
        embinfo("Input ids: %s", self.input_ids)
        with torch.no_grad():
            embeddings = self.forward(self.input_ids)
        cur_embeds = weight[self.prompt_ids,:].detach()
        embinfo("cur embeddings: %s", cur_embeds)
        new_embeds = embeddings.detach()
        weight[self.prompt_ids,:]=new_embeds 
        embinfo("%%%%%%%%%%%%%%%%%%%%%%%%%% dump embeddings end %%%%%%%%%%%%%%%%%%")



if __name__ == "__main__":
    #Unit Test
    ## 3. T5
    model = transformers.T5ForConditionalGeneration.from_pretrained("/home/ahmad/pret/t5-small/")
    tokenizer = transformers.T5Tokenizer.from_pretrained("/home/ahmad/pret/t5-small/")
    print(f"Test T5")
    print(f"Original tokenizer size: {len(tokenizer)}")
    #wrapped_model,prompt_func,prompt_string = PTuningWrapper.\
    #    interval_prompt(
    #        model,tokenizer,(2,3),(1,1),return_prompt_string=True
    #    )
    length=5
    embedding_dim = 512
    id_offset = len(tokenizer)
    prompt_encoder = None
    decoder_prompt_encoder = None
    prompt_encoder = LSTMEmbeddingPromptEncoder(length,embedding_dim,id_offset)
    decoder_prompt_encoder = LSTMEmbeddingPromptEncoder(length,embedding_dim,id_offset)
    rel="xIntent"
    def get_prompt_token_fn(id_offset,length):
        return lambda x: (x>=id_offset)&(x<id_offset+length)
    added_tokens = [ 
        transformers.AddedToken(f"<{rel}_{i}>",lstrip=True,
            rstrip=False)
        for i in 
            range(length)
    ]
    tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
    model.resize_token_embeddings(len(tokenizer))
    wrapped_model = PTuningWrapper(model,prompt_encoder,decoder_prompt_encoder,prompt_token_fn=get_prompt_token_fn(id_offset,length))
    print(f"Tokenizer size: {len(tokenizer)}")
    example_encoder = "<xIntent_0> <xIntent_1> This is <xIntent_2>"
    tokenized = tokenizer(example_encoder,return_tensors="pt")
    example_decoder = "<xIntent_3> <xIntent_4> This is <xIntent_5>"
    tokenized_decoder_labels = tokenizer(example_decoder,return_tensors="pt")
    tokenized['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(
        tokenized_decoder_labels['input_ids']
    )
    tokenized['labels'] = tokenized['input_ids'] #mycode
    tokenized['decoder_attention_mask'] = tokenized_decoder_labels['attention_mask']
    #tokenized["decoder_input_ids"] = tokenized_decoder_labels["input_ids"]
    print("Tokenized:",tokenized)
    for p in model.parameters():
        p.requires_grad = False
    wrapped_model.zero_grad()
    results = wrapped_model(**tokenized)
    #print("Forward Results:", results)
    print("Try backward")
    loss = torch.sum(results[0])
    loss.backward()
    print("Original embedding grads:",model.get_input_embeddings().weight.grad)
    if wrapped_model.prompt_encoder:
        print("Prompt embedding grads:", wrapped_model.prompt_encoder.embedding.weight.grad)
    wrapped_model.update_model_weight()
    #print(model.get_input_embeddings().weight[wrapped_model.decoder_prompt_encoder.id_offset]==\
    #    wrapped_model.decoder_prompt_encoder.forward(torch.tensor([wrapped_model.decoder_prompt_encoder.id_offset])))
    import sys
    sys.exit() 
    ## 1. BERT
    model = transformers.BertModel.from_pretrained("/drive2/pretrained/bert/parsbert/bert-base-parsbert-uncased")
    tokenizer = transformers.BertTokenizer.from_pretrained("/drive2/pretrained/bert/parsbert/bert-base-parsbert-uncased/")
    print(f"Test BERT")
    print(f"Original tokenizer size: {len(tokenizer)}")
    wrapped_model,prompt_func,prompt_string = PTuningWrapper.\
        interval_prompt(
            model,tokenizer,(2,3,1),return_prompt_string=True
        )
    print(f"Prompt length:{wrapped_model.prompt_encoder.length}")
    print(f"Tokenizer size: {len(tokenizer)}")
    print("Prompt string:",prompt_string)
    example = prompt_func("piece one","piece two")
    print("Example:",example)
    tokenized = tokenizer(example,return_tensors="pt")
    print("Tokenized:",tokenized)
    for p in model.parameters():
        p.requires_grad = False
    wrapped_model.zero_grad()
    out = wrapped_model(**tokenized)
    print("Try backward")
    loss = torch.sum(out[1])
    loss.backward()
    print("Original embedding grads:",model.get_input_embeddings().weight.grad)
    print("Prompt embedding grads:", wrapped_model.prompt_encoder.embedding.weight.grad)
    ## 2. GPT2
    model = transformers.GPT2Model.from_pretrained("/drive2/pretrained/gpt/gpt2/")
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("/drive2/pretrained/gpt/gpt2/")
    print(f"Test GPT2")
    print(f"Original tokenizer size: {len(tokenizer)}")
    wrapped_model,prompt_func,prompt_string = PTuningWrapper.\
        interval_prompt(
            model,tokenizer,(2,3,1),return_prompt_string=True
        )
    print(f"Prompt length:{wrapped_model.prompt_encoder.length}")
    print(f"Tokenizer size: {len(tokenizer)}")
    print("Prompt string:",prompt_string)
    example = prompt_func("piece one","piece two")
    print("Example:",example)
    tokenized = tokenizer(example,return_tensors="pt")
    print("Tokenized:",tokenized)
    for p in model.parameters():
        p.requires_grad = False
    wrapped_model.zero_grad()
    out = wrapped_model(**tokenized)
    print("Try backward")
    loss = torch.sum(out[0])
    loss.backward()
    print("Original embedding grads:",model.get_input_embeddings().weight.grad)
    print("Prompt embedding grads:", wrapped_model.prompt_encoder.embedding.weight.grad)



