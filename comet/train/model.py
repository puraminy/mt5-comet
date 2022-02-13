from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast, 
    AutoModelForSeq2SeqLM, 
    MT5ForConditionalGeneration, MT5TokenizerFast, AdamW, AddedToken,
    GPT2LMHeadModel, GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

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

