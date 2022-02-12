from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast, 
    AutoModelForSeq2SeqLM, 
    MT5ForConditionalGeneration, MT5TokenizerFast, AdamW, AddedToken,
    GPT2LMHeadModel, GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

model = T5ForConditionalGeneration.from_pretrained("/home/pouramini/pret/t5-base/")
print(model)
# All modules in the 
modules_to_freeze = [model.encoder.block[i].layer[0].SelfAttention.k for i in range(len(model.encoder.block))]
modules_to_freeze.extend([model.encoder.block[i].layer[0].SelfAttention.v for i in range(len(model.encoder.block))])
# And the decoder modules, which has both a SelfAttention (layer[0]) 
modules_to_freeze.extend([model.decoder.block[i].layer[0] for i in range(len(model.decoder.block))])
# and CrossAttention (layer[1]) block
modules_to_freeze.extend([model.decoder.block[i].layer[1] for i in range(len(model.decoder.block))])

for module in modules_to_freeze:
    for param in module.parameters():
        param.requires_grad = False  # Actual freezing operation

for param in model.parameters():
    print(param.name, "---", param.requires_grad)

