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
    def __init__(self, para1, para2, lr1, lr2):
        self.optimizer1 = AdamW(para1, lr=lr1, betas=(0.9, 0.999))
        self.optimizer2 = AdamW(para2, lr=lr2, betas=(0.9, 0.999))

    def step(self):
        self.optimizer1.step()
        self.optimizer2.step()

    def zero_grad(self):
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

    def state_dict(self):
        return {
            'optimizer1': self.optimizer1.state_dict(),
            'optimizer2': self.optimizer2.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.optimizer1.load_state_dict(state_dict['optimizer1'])
        self.optimizer2.load_state_dict(state_dict['optimizer2'])

    def cuda(self):
        for state in self.optimizer1.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in self.optimizer2.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

class Scheduler:
    def __init__(self, optim, step1=10000, step2=10000, gamma1=.1, gamma2=.1):
        self.scheduler1 = torch.optim.lr_scheduler.StepLR(optim.optimizer1, step1, gamma1)
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(optim.optimizer2, step2, gamma2)

    def step(self):
        self.scheduler1.step()
        self.scheduler2.step()


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

