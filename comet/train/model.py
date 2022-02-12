from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast, 
    AutoModelForSeq2SeqLM, 
    MT5ForConditionalGeneration, MT5TokenizerFast, AdamW, AddedToken,
    GPT2LMHeadModel, GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

def freeze_self_att(model_to_freeze, part, ed):
    if "k" in part:
        modules_to_freeze.extend([ed.block[i].layer[0].SelfAttention.k for i in range(len(model.encoder.block))])
    if "v" in part:
        modules_to_freeze.extend([ed.block[i].layer[0].SelfAttention.v for i in range(len(model.encoder.block))])
    if "q" in part:
        modules_to_freeze.extend([ed.block[i].layer[0].SelfAttention.q for i in range(len(model.encoder.block))])
    if "o" in part:
        modules_to_freeze.extend([ed.block[i].layer[0].SelfAttention.o for i in range(len(model.encoder.block))])
    if "f" in part:
        modules_to_freeze.extend([ed.block[i].layer[0].T5LayerFF for i in range(len(model.encoder.block))])

def freeze_cross_att(model_to_freeze, part, ed):
    if "k" in part:
        modules_to_freeze.extend([ed.block[i].layer[1].EncDecAttention.k for i in range(len(model.encoder.block))])
    if "v" in part:
        modules_to_freeze.extend([ed.block[i].layer[1].EncDecAttention.v for i in range(len(model.encoder.block))])
    if "q" in part:
        modules_to_freeze.extend([ed.block[i].layer[1].EncDecAttention.q for i in range(len(model.encoder.block))])
    if "o" in part:
        modules_to_freeze.extend([ed.block[i].layer[1].EncDecAttention.o for i in range(len(model.encoder.block))])

