#暂时没考虑encoder和decoder的tokenizer不同的情况，以后可以给decoder全套的prompt_token_fn
import wandb
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
from transformers.optimization import Adafactor, AdafactorSchedule, AdamW
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

wargs = {}

def set_wargs(args):
    global wargs 
    wargs =args

home = expanduser("~")
wlog = logging.getLogger("comet.wrapper")
emblog = logging.getLogger("comet.embedding")
consoleHandler = logging.StreamHandler()
#wlog.addHandler(consoleHandler)
#emblog.addHandler(consoleHandler)
tlog = logging.getLogger("comet.time")
FORMAT = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)10s() ] %(message)s")

def tinfo(text, *args, **kwargs):
    tlog.info(text, *args)

def winfo(text, *args, **kwargs):
    wlog.info(text, *args)
    #print((text, *args))

def embinfo(text, *args, **kwargs):
    emblog.info(text, *args)
    #print((text, *args))

def getFname(name, path=""):
    if not path:
        if "ahmad" in home or "pouramini" in home:
            path = os.path.join(home, "mt5-comet", "comet", "output")
        else:
            path = "/content"
    logFilename = os.path.join(path, f"{name}.log")
    return logFilename

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

        set_wargs(args)
        wHandler = logging.FileHandler(getFname(args["rel_filter"] + "_wrapper"), mode='w')
        wHandler.setFormatter(FORMAT)
        wlog.addHandler(wHandler)
        eHandler = logging.FileHandler(getFname(args["rel_filter"] + "_embedding"), mode='w')
        eHandler.setFormatter(FORMAT)
        emblog.addHandler(eHandler)
        tHandler = logging.FileHandler(getFname(args["rel_filter"] + "_time", 
            path=args["save_path"]), mode='w')
        tHandler.setFormatter(FORMAT)
        tlog.addHandler(tHandler)
        embinfo("Embedding log")
        winfo("Wrapper log")
        wlog.setLevel(logging.INFO)
        emblog.setLevel(logging.INFO)
        tlog.setLevel(logging.INFO)


        super().__init__()
        mbp("")
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
            

    def add_prompt_encoder(self, encoder):
        self.prompt_encoders.append(encoder)

    def encoder_forward(self, encoder, input_ids, inputs_embeds, tids):
        #encoder = self.prompt_encoders[0]
        device=inputs_embeds.device
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
            prompt_embeds = encoder(prompt_input_ids, tids).to(device)
            winfo("Prompt Embeds size: %s", prompt_embeds.size())
            winfo("Encoder mask: %s", encoder_masks.size())
            inputs_embeds[encoder_masks]=prompt_embeds
            return prompt_embeds
        return None

    #def generate(self, task_ids, *args, **kwargs):
        #inform_layers(self.underlying_model, self.adapter_class, task_ids)
    #    return self.underlying_model.generate(*args, **kwargs)

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
    def update_model_weight(self):
        winfo(f"Updating model weights")
        self.cur_embeddings = self.underlying_model.get_input_embeddings()
        if self.merge_encoder:
            self.merge_encoder.dump_embeddings_into(self.cur_embeddings.weight)
        elif self.flat_encoder:
            self.flat_encoder.dump_embeddings_into(self.cur_embeddings.weight)
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
                winfo(f"the wrapper has prompt encoder for decoder part")
                # fill the current embeddings with weights of encoder
                encoder.dump_embeddings_into(
                                       self.model_decoder_embeddings.weight)
                #self.prompt_encoder.dump_embeddings_into(self.model_embeddings.weight)
        if self.decoder_prompt_encoder:
            self.decoder_prompt_encoder.dump_embeddings_into(
                self.model_decoder_embeddings.weight)


class PromptEncoder(torch.nn.Module):
    def __init__(self,name, length,embedding_dim,id_offset, init_embs, prompt_ids, lr=0.01,**kwargs) -> None:
        super().__init__()
        self.learning_rate = lr
        self.length = length
        self.name = name
        self.device = "cpu"
        self.counter = 0
        self.prompt_ids = prompt_ids
        self.input_ids = torch.nn.parameter.Parameter(torch.tensor(prompt_ids),
             requires_grad=False)
        embinfo("=========================== %s ===================", name)
        embinfo("prompt ids: %s", prompt_ids)
        self.embedding_dim = embedding_dim
        self.id_offset = id_offset
        self.embedding = torch.nn.Embedding(length,embedding_dim)
        self.init_embs = init_embs
        #self.embedding.weight.requires_grad = False
        if init_embs:
            with torch.no_grad():
                for _id,emb in init_embs.items():
                    if _id < len(self.embedding.weight):
                        self.embedding.weight[_id] = emb
                        embinfo("%s : %s", _id, emb)
        para = [p for p in self.parameters() if p.requires_grad ]
        self.optimizer = AdamW(para, lr=self.learning_rate, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 10000, 0.1)

    def isin(self, ar1, ar2):
        return (ar1[..., None] == ar2).any(-1)
    def get_prompt_token_fn(self):
        if self.input_ids is not None:
            return lambda x: self.isin(x, self.input_ids)
        else:
            return lambda x: (x>=self.id_offset)&(x<self.id_offset+self.length)
    def dump_embeddings_into(self,weight):
        raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
        self.scheduler.step()

class EmbeddingPromptEncoder(PromptEncoder):
    def forward(self,prompt_token_ids,pids=None):
        if self.id_offset > 0:
            prompt_token_ids = prompt_token_ids - self.id_offset
        else:
            prompt_token_ids = (prompt_token_ids.view(-1,1) == self.input_ids).int().argmax(dim=1)
        ret_embs = self.embedding(prompt_token_ids)
        return ret_embs

    def dump_embeddings_into(self, weight):
        detached_embeddings = self.embedding.weight.detach()
        weight[self.prompt_ids,:]=detached_embeddings

class MatPromptEncoder(PromptEncoder):
    def __init__(self, prefix_config, **kwargs):
        super().__init__(**kwargs)
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

            bound = 1 / math.sqrt(prefix_config['n_prompt_tokens'] * self.embedding_dim)
            self.A = nn.Parameter(data=torch.empty((
                prefix_config['intrinsic_dim'],
                prefix_config['n_prompt_tokens'] * self.embedding_dim
            )).uniform_(-bound, bound), requires_grad=False)

    def forward(self, prompt_token_ids, tids=None, training=True):
        if self.id_offset > 0:
            index_list = prompt_token_ids - self.id_offset
        else:
            index_list = (prompt_token_ids.view(-1,1) == self.input_ids).int().argmax(dim=1)
        router = self.router[0] # torch.index_select(self.router, 0, tids)
        if training:
            router = RelaxedBernoulli(temperature=self.temperature, logits=router).rsample()  # layer * n_prompts
        else:
            router = torch.sigmoid(router)  # layer * n_prompts
        router = (router / (router.sum(dim=-1, keepdim=True) + 1e-12))  # layer * 1 * n_prompts
        z = torch.mm(self.z, self.A) if not hasattr(self, 'prompt') else self.prompt
        #ret_embeds = torch.matmul(router.unsqueeze(0), z).view(-1, self.embedding_dim)
        running_weight = torch.matmul(router, z).view(-1, self.embedding_dim)
        ret_embeds = F.embedding(index_list, running_weight)
        return ret_embeds 

    def dump_embeddings_into(self, weight):
        with torch.no_grad():
            embs = self.forward(self.input_ids, training=False)
        detached_embeddings = embs.detach()
        weight[self.prompt_ids,:]=detached_embeddings
        


class MergePromptEncoder(PromptEncoder):
    def __init__(self, encoders = [], trunc_router=False, wandb=False, **kwargs):
        super().__init__(**kwargs)
        self.task_id = 0
        self.temperature = 1 
        self.n_prompts = 5 #len(encoders) 
        self.n_tasks = 2
        self.flag = True
        self.trunc_router = trunc_router
        self.wandb = wandb
        if encoders:
            self.encoders = torch.nn.ModuleList(encoders)
        self.router = nn.Parameter(data=torch.empty((
            self.n_tasks,
            self.n_prompts
        )).uniform_(-1e-3, 1e-3))

    def forward(self, prompt_token_ids, tids=None, training=True):
        device = self.device
        task_id = tids[0]
        if self.wandb:
            wandb.log({'tid': task_id})
        if self.flag:
            tinfo("Initial Router: %s", self.router)
            self.flag = False
        router = self.router[task_id]
        if training:
            router = RelaxedBernoulli(temperature=self.temperature, logits=router).rsample()  # layer * n_prompts
            router = (router / (router.sum(dim=-1, keepdim=True) + 1e-12))  
        else:
            #router = torch.sigmoid(router)  # layer * n_prompts
            if self.trunc_router:
                with torch.no_grad():
                    tinfo("Router:========================================")
                    tinfo("Router Before relu: %s", router)
                    router[router <= 0] = 0
                    router[router > 0] = 1
                    tinfo("Router After relu: %s", router)
            #router = (router / (router.sum(dim=-1, keepdim=True) + 1e-12))  
        # layer * 1 * n_prompts
        #ret_embeds = torch.matmul(router.unsqueeze(0), z).view(-1, self.embedding_dim)
        if self.id_offset > 0:
            index_list = prompt_token_ids - self.id_offset
        else:
            index_list = (prompt_token_ids.view(-1,1) == self.input_ids).int().argmax(dim=1)
        z = torch.zeros(len(self.prompt_ids), self.embedding_dim).to(device)
        tl = []
        for encoder in self.encoders:
            pids = encoder.input_ids
            out = encoder(pids).to(device) 
            tl.append(out)
        z = torch.vstack(tl) 
        z = z.view(len(self.encoders), -1) 
        running_weight = torch.matmul(router, z).view(-1, self.embedding_dim)
        ret_embeds = F.embedding(index_list, running_weight)
        return ret_embeds 

    def dump_embeddings_into(self, weight):
        tinfo("Final Router (ReLU): %s", self.router)
        with torch.no_grad():
            embs= self.forward(self.input_ids,tids=[0], training=False)
            embs2= self.forward(self.input_ids,tids=[1], training=False)
            #embs = embs1 + embs2
        tinfo("Router After Forward: %s", self.router)
        detached_embeddings = embs.detach()
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
        if self.id_offset > 0:
            prompt_token_ids = prompt_token_ids - self.id_offset
        else:
            prompt_token_ids = (prompt_token_ids.view(-1,1) == self.input_ids).int().argmax(dim=1)
        embs = self.embedding(prompt_token_ids)
        ret_embs = self.mlp(embs)
        return ret_embs

    def dump_embeddings_into(self, weight):
        with torch.no_grad():
            embs = self.forward(self.input_ids)
        detached_embeddings = embs.detach()
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
        net_inputs = self.net_inps
        if self.id_offset > 0:
            net_inputs = self.input_ids - self.id_offset
            #index_list = [((net_inputs == x).nonzero(as_tuple=True)[0]) for x in prompt_token_ids_2]
        index_list = (prompt_token_ids.view(-1,1) == self.input_ids).int().argmax(dim=1)
        # create embedding vectors for input ids
        embeds = self.embedding(net_inputs)
        x = self.lstm(embeds.unsqueeze(0))
        running_weight = self.mlp(x[0]).squeeze(0)
        ret_embeds = F.embedding(index_list,running_weight)
        return ret_embeds

    def dump_embeddings_into(self, weight):
        with torch.no_grad():
            embeddings = self.forward(self.input_ids)
        cur_embeds = weight[self.prompt_ids,:].detach()
        new_embeds = embeddings.detach()
        weight[self.prompt_ids,:]=new_embeds 


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



