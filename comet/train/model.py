import torch
from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast, 
    AutoModelForSeq2SeqLM, 
    MT5ForConditionalGeneration, MT5TokenizerFast, AdamW, AddedToken,
    GPT2LMHeadModel, GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

class Optim:
    def __init__(self, paras, lrs):
        self.opts = []
        for para,lr in zip(paras,lrs):
            opt = AdamW(para, lr=lr, betas=(0.9, 0.999))
            self.opts.append(opt)

    def step(self):
        for opt in self.opts:
            self.opt.step()

    def zero_grad(self):
        for opt in self.opts:
            opt.zero_grad()

    def state_dict(self):
        ret = {}
        for i, opt in enumerate(self.opts):
            ret['opt'+ str(i)] = opt.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        for i, opt in enumerate(self.opts):
            opt.load_state_dict(state_dict['opt'+ str(i)])

    def cuda(self):
        for opt in self.opts:
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

class Scheduler:
    def __init__(self, optim, step=10000, gamma=.1):
        self.schedulers = []
        for opt in optim.opts:
            self.schedulers.append(torch.optim.lr_scheduler.StepLR(opt, step, gamma))

    def step(self):
        for sch in self.schedulers:
           sch.step()

def freeze_self_att(modules_to_freeze, part, ed, decoder=False):
    if part == "none":
       return
    fi = 1
    if decoder: 
        fi = 2
    if "k" in part:
        modules_to_freeze.extend([ed.block[i].layer[0].SelfAttention.k for i in range(12)])
    if "v" in part:
        modules_to_freeze.extend([ed.block[i].layer[0].SelfAttention.v for i in range(12)])
    if "q" in part:
        modules_to_freeze.extend([ed.block[i].layer[0].SelfAttention.q for i in range(12)])
    if "o" in part:
        modules_to_freeze.extend([ed.block[i].layer[0].SelfAttention.o for i in range(12)])
    if "f" in part or "wi" in part:
        modules_to_freeze.extend([ed.block[i].layer[fi].DenseReluDense.wi for i in range(12)])
    if "f" in part or "wo" in part:
        modules_to_freeze.extend([ed.block[i].layer[fi].DenseReluDense.wo for i in range(12)])

def freeze_cross_att(modules_to_freeze, part, ed):
    if part == "none":
       return
    if "k" in part:
        modules_to_freeze.extend([ed.block[i].layer[1].EncDecAttention.k for i in range(12)])
    if "v" in part:
        modules_to_freeze.extend([ed.block[i].layer[1].EncDecAttention.v for i in range(12)])
    if "q" in part:
        modules_to_freeze.extend([ed.block[i].layer[1].EncDecAttention.q for i in range(12)])
    if "o" in part:
        modules_to_freeze.extend([ed.block[i].layer[1].EncDecAttention.o for i in range(12)])

