#暂时没考虑encoder和decoder的tokenizer不同的情况，以后可以给decoder全套的prompt_token_fn

import re
from pathlib import Path
import transformers
import numpy as np
import torch
import torch.nn.functional as F
def _isin(tensor:torch.Tensor,values:torch.Tensor):
    return (tensor[..., None] == values).any(-1)
import logging
import os
from os.path import expanduser
home = expanduser("~")
wlog = logging.getLogger("comet.wrapper")
emblog = logging.getLogger("comet.embedding")
FORMAT = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)10s() ] %(message)s")
def getFname(name):
    if "ahmad" in home or "pouramini" in home:
        logFilename = os.path.join(home, f"logs/{name}.log")
    else:
        logFilename = f"/content/{name}.log"
    return logFilename
wHandler = logging.FileHandler(getFname("wrapper"), mode='w')
wHandler.setFormatter(FORMAT)
wlog.addHandler(wHandler)
eHandler = logging.FileHandler(getFname("embedding"), mode='w')
eHandler.setFormatter(FORMAT)
emblog.addHandler(eHandler)
emblog.info("Embedding log")
wlog.info("Wrapper log")
wlog.setLevel(logging.INFO)
emblog.setLevel(logging.INFO)

class PTuningWrapper(torch.nn.Module):
    def __init__(self,model,prompt_encoders,decoder_prompt_encoder=None,
        prompt_token_fn=None, prompt_token_id=None, prompt_token_ids=None,
        replacing_token_id=0, do_log=True, merge_prompts = False):
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
        wlog.disabled = not do_log
        self.ll = logging.INFO
        wlog.info("%%%%%%%%%%%%%%%%%%%%%%%% New version %%%%%%%%%%%%%%%%%%")
        self.underlying_model = model
        self.model_embeddings = model.get_input_embeddings()
        wlog.info("self.model embedding:{}".format(self.model_embeddings))
        model_embeddings_size = model.get_input_embeddings().num_embeddings
        wlog.info("model embedding_size:{}".format(model_embeddings_size))
        self.prompt_encoders = torch.nn.ModuleList(prompt_encoders)
        wlog.info("num of encoders %s:", len(self.prompt_encoders))
        self.merge_prompt_ids = []
        sum_len = 0
        for encoder in self.prompt_encoders:
            _ids = encoder.prompt_ids
            sum_len += len(_ids)
            for _id in _ids:
                if not _id in self.merge_prompt_ids:
                    self.merge_prompt_ids.append(_id)

            _offset = min(_ids)
            wlog.info("** existing encoder ids for %s: %s", encoder.name, _ids)
            rel_ids_tensor = torch.LongTensor(_ids)
            embs = self.model_embeddings
            rel_embs = embs(rel_ids_tensor)
            with torch.no_grad():
               for i, e in enumerate(rel_embs):
                   encoder.embedding.weight[i] = e #.detach()

        self.embedding_dim = model.config.hidden_size
        self.merge_offset = min(self.merge_prompt_ids)
        wlog.info("Merge ids: %s,", self.merge_prompt_ids)
        wlog.info("Merge ids len: %s,", len(self.merge_prompt_ids))
        wlog.info("Sum len: %s,", sum_len)
        wlog.info("Offset: %s,", self.merge_offset)

        self.merge_encoder = None 
        self.merge_embedding = None
        if merge_prompts:
            self.merge_encoder = None #LSTMEmbeddingPromptEncoder("wrap_all", len(self.merge_prompt_ids), self.embedding_dim, self.merge_offset, prompt_ids=self.merge_prompt_ids)
            self.merge_embedding = torch.nn.Embedding(len(self.merge_prompt_ids), self.embedding_dim)
            for encoder in self.prompt_encoders:
                encoder.embedding = self.merge_embedding
                encoder.id_offset= self.merge_offset
                encoder.length= len(self.merge_prompt_ids)


        self.decoder_prompt_encoder = decoder_prompt_encoder
        self.replacing_token_id = replacing_token_id
        wlog.info("REP id:{}".format(replacing_token_id))
        if self.decoder_prompt_encoder:
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
            
            #Default: All token ids beyond num_embeddings are seen as prompt token
    def add_prompt_encoder(self, encoder):
        self.prompt_encoders.append(encoder)

    def forward(self,input_ids, labels, decoder_input_ids=None,pids=None,**kwargs):
        ll = self.ll # log level
        wlog.log(ll, "wrapper forward was called")
        wlog.log(ll, "Prompt ids:{}".format(pids))
        # find masks based on the range of prompt ids (offset_id < X < offset_id + prompt_length)
        #Because this wrapper only deals with a single prompt, the length should be the same, you can use masked_select to reshape 
        prompt_masks = self.prompt_token_fn(input_ids)
        if prompt_masks.any():
            wlog.log(ll, "promp masks:{}".format(prompt_masks))
            input_ids_ = input_ids.clone()
            wlog.info("inpu ids :{}".format(input_ids))
            if self.replacing_token_id is not None:
                # replace prompt ids in input_ids with replacing token
                input_ids_[prompt_masks]=self.replacing_token_id
            # find the model embeddings of input ids except for prompt tokens
            inputs_embeds = self.model_embeddings(input_ids_)
            device=inputs_embeds.device
            all_prompts_input_ids = input_ids[prompt_masks]
            wlog.info("All prompts input ids: %s", all_prompts_input_ids)
            wlog.info("Len All prompts input ids: %s", len(all_prompts_input_ids))
            if self.merge_encoder:
                prompt_embds = self.merge_encoder(all_prompts_input_ids,pids).to(device)
                inputs_embeds[prompt_masks]=prompt_embds
            else:
                embeds_list = []
                for encoder in self.prompt_encoders:
                    #encoder = self.prompt_encoders[0]
                    wlog.info("********** offset: %s, length: %s", encoder.id_offset, encoder.length)
                    prompt_token_fn = encoder.get_prompt_token_fn()
                    encoder_masks = prompt_token_fn(input_ids)
                    wlog.info("Encoder masks: %s", encoder_masks)
                    if encoder_masks.any():
                        #find input ids for prompt tokens
                        prompt_input_ids = input_ids[encoder_masks]
                        wlog.info("Prompt Input ids: %s", prompt_input_ids)
                        # call forwards on prompt encoder whose outputs are prompt embeddings
                        prompt_embeds = encoder(prompt_input_ids,\
                            pids).to(device)
                        embeds_list.append(prompt_embeds)
                        # replace prompt_embeddings calculated by prompt encoder in input embeddings
                        wlog.info("Prompt Embeds: %s", prompt_embeds)
                        inputs_embeds[encoder_masks]=prompt_embeds
                cat = torch.cat(embeds_list)
                wlog.info("CAT Embeds: %s", cat)
                wlog.info("CAT Embeds size: %s", cat.size())
        else:
            inputs_embeds = self.model_embeddings(input_ids)
        
        if decoder_input_ids is not None:
            if self.decoder_prompt_encoder is not None:
                wlog.log(ll, "prompt decoder exists")
                decoder_prompt_masks = self.prompt_token_fn(decoder_input_ids)
                if decoder_prompt_masks.any():
                    wlog.log(ll,"decoder prompt mask:{}".format(decoder_prompt_masks))
                    wlog.log(ll,f"decoder mask, replacing token id: {self.replacing_token_id}")
                    decoder_input_ids_ = decoder_input_ids.clone()
                    if self.replacing_token_id is not None:
                        decoder_input_ids_[decoder_prompt_masks] = \
                            self.replacing_token_id
                    decoder_inputs_embeds = self.model_decoder_embeddings(
                        decoder_input_ids_
                    )
                    wlog.log(ll,f"decoder input ids {decoder_input_ids}")
                    decoder_prompt_embeds = self.decoder_prompt_encoder(
                        decoder_input_ids[decoder_prompt_masks],pids).to\
                        (device=decoder_inputs_embeds.device)
                    decoder_inputs_embeds[decoder_prompt_masks] = \
                        decoder_prompt_embeds
                    decoder_labels_masks = self.prompt_token_fn(labels)
                    labels[decoder_labels_masks] = 0
                    wlog.log(ll, labels)
                else:
                    decoder_inputs_embeds = self.model_decoder_embeddings(
                        decoder_input_ids
                    )
                return self.underlying_model(inputs_embeds=inputs_embeds,
                    decoder_inputs_embeds=decoder_inputs_embeds, labels=labels,**kwargs
                )
            else: #decoder_prompt_encoder is not defined, so decoder_originical_embedding is not set.
                self.ll = logging.DEBUG
                return self.underlying_model(inputs_embeds=inputs_embeds,
                    decoder_input_ids=decoder_input_ids, labels=labels,**kwargs)
        else:
            self.ll = logging.DEBUG
            return self.underlying_model(inputs_embeds=inputs_embeds,**kwargs)
    def update_model_weight(self):
        wlog.info(f"Updating model weights")
        self.cur_embeddings = self.underlying_model.get_input_embeddings()
        if self.merge_encoder:
            self.merge_encoder.dump_embedding(self.cur_embeddings.weight)
        elif self.merge_embedding:
            detached_embeddings = self.merge_embedding.weight.detach()
            emblog.info("Dump embeddings merge_embeddings: %s", detached_embeddings)
            self.cur_embeddings.weight[self.merge_prompt_ids,:]=detached_embeddings
        else:
            for encoder in self.prompt_encoders:
                wlog.info(f"the wrapper has prompt encoder")
                # fill the current embeddings with weights of encoder
                encoder.dump_embedding(self.cur_embeddings.weight)
                #self.prompt_encoder.dump_embedding(self.model_embeddings.weight)
        if self.decoder_prompt_encoder in self.prompt_encoders:
            wlog.info(f"Encoder and Decoder are the same")
            pass
        elif self.decoder_prompt_encoder:
            self.decoder_prompt_encoder.dump_embedding(
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
        emblog.info("=========================== %s ===================", name)
        emblog.info("prompt ids: %s", prompt_ids)
        self.embedding_dim = embedding_dim
        self.id_offset = id_offset
        self.embedding = torch.nn.Embedding(length,embedding_dim)
        #self.embedding.weight.requires_grad = False
        if init_embs:
            with torch.no_grad():
                for _id,emb in init_embs.items():
                    if _id < len(self.embedding.weight):
                        self.embedding.weight[_id] = emb
                        emblog.info("%s : %s", _id, emb)

    def isin(self, ar1, ar2):
        return (ar1[..., None] == ar2).any(-1)
    def get_prompt_token_fn(self):
        if self.input_ids is not None:
            return lambda x: self.isin(x, self.input_ids)
        else:
            return lambda x: (x>=self.id_offset)&(x<self.id_offset+self.length)
    def dump_embedding(self,weight):
        raise NotImplementedError


class EmbeddingPromptEncoder(PromptEncoder):
    def __init__(self,name,length,embedding_dim,id_offset,init_embs=None, prompt_ids=[]) -> None:
        super().__init__(name, length,embedding_dim,id_offset, init_embs, prompt_ids)
    
    def forward(self,prompt_token_ids,pids=None):
        emblog.info("=========================== Forward ===================")
        emblog.info("=========================== %s ===================", self.name)
        emblog.info("Before prompt token ids: %s", prompt_token_ids)
        #emblog.info("id offset: %s", self.id_offset)
        #emblog.info("id length: %s", self.length)
        if self.id_offset > 0:
            prompt_token_ids = prompt_token_ids - self.id_offset
        else:
            prompt_token_ids = (prompt_token_ids.view(-1,1) == self.input_ids).int().argmax(dim=1)
        emblog.info("self input ids: %s", self.input_ids)
        emblog.info("After prompt token ids: %s", prompt_token_ids)
        emblog.info(self.embedding.weight)
        ret_embs = self.embedding(prompt_token_ids)
        emblog.info("ret embs %s", ret_embs)
        emblog.info("=========================== Forward end ===================")
        return ret_embs

    def dump_embedding(self, weight):
        wlog.info("Dump embeddings")
        emblog.info("=========================== %s ===================", self.name)
        emblog.info("input weights: %s", weight)
        detached_embeddings = self.embedding.weight.detach()
        emblog.info("Dump embeddings: %s", detached_embeddings)
        emblog.info("on this ids: %s", self.prompt_ids)
        weight[self.prompt_ids,:]=detached_embeddings


class LSTMEmbeddingPromptEncoder(PromptEncoder):
    def __init__(self,name, length,embedding_dim,id_offset, init_embs=None, prompt_ids=[]) -> None:
        super().__init__(name, length,embedding_dim,id_offset, init_embs, prompt_ids)
        self.net_inps = torch.nn.parameter.Parameter(torch.arange(length),
            requires_grad=False)
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim //2, #my code
            num_layers=2,
            dropout=0,
            bidirectional=True,
            batch_first=True
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, embedding_dim)
        )


    def forward(self,prompt_token_ids,pids=None):
        emblog.info("=========================== Forward begin ===================")
        emblog.info("=========================== %s ===================", self.name)
        emblog.info("before prompt token ids:{}".format(prompt_token_ids))
        emblog.info("self input ids: %s", self.input_ids)
        emblog.info("NETTTTT inps:{}".format(self.net_inps))
        # find zero based ids 
        net_inputs = self.net_inps
        if self.id_offset > 0:
            prompt_token_ids = prompt_token_ids - self.id_offset
            net_inputs = self.input_ids - self.id_offset
            index_list = [((net_inputs == x).nonzero(as_tuple=True)[0]) for x in prompt_token_ids]
            index_list = torch.tensor(index_list)
        else:
            index_list = (prompt_token_ids.view(-1,1) == self.input_ids).int().argmax(dim=1)
        emblog.info("after prompt token ids:  %s", prompt_token_ids)
        emblog.info("after net inputs:  %s", net_inputs)
        emblog.info("index list:  %s", index_list)
        # create embedding vectors for input ids
        embeds = self.embedding(net_inputs)
        # do forward calculations
        x = self.lstm(embeds.unsqueeze(0))
        emblog.info("XXXXXXXXXXXXXXXXX: %s",x)
        emblog.info("XXXXXXXXXXXXXXXXX[0]: %s",x[0])
        emblog.info("XXXXXXXXXXXXXXXXX size: %s",x[0].size())
        emblog.info("lstml embeds: %s",embeds)

        running_weight = self.mlp(x[0]).squeeze(0)
        if self.counter < 5:
            emblog.info("--------------------")
            emblog.info("running weights: %s",running_weight)
            emblog.info("running weights size: %s",running_weight.size())
            self.counter += 1

        # return weights for prompt_token_ids 
        ret_embeds = F.embedding(index_list,running_weight)
        emblog.info("ret embeds size %s", ret_embeds.size())
        emblog.info("ret embeds %s", ret_embeds)
        emblog.info("=========================== Forward end ===================")
        return ret_embeds
    def dump_embedding(self, weight):
        # get embedding weights as the output of forward pass
        emblog.info("%%%%%%%%%%%%%%%%%%%%%%%%%% dump embeddings start %%%%%%%%%%%%%%%%")
        emblog.info("=========================== %s ===================", self.name)
        emblog.info("Dump embeddings: %s", weight)
        emblog.info("Input ids: %s", self.input_ids)
        with torch.no_grad():
            embeddings = self.forward(self.input_ids)
        cur_embeds = weight[self.prompt_ids,:].detach()
        emblog.info("cur embeddings: %s", cur_embeds)
        new_embeds = embeddings.detach()
        emblog.info("new embeddings: %s", new_embeds)
        mean_embeds = (cur_embeds + new_embeds)/2
        emblog.info("replace embeddings: %s", mean_embeds)
        weight[self.prompt_ids,:]=new_embeds # mean_embeds
        emblog.info("%%%%%%%%%%%%%%%%%%%%%%%%%% dump embeddings end %%%%%%%%%%%%%%%%%%")



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

