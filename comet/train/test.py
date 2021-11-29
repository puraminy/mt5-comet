from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast,AutoTokenizer,
    )

underlying_model_name = "/home/ahmad/pret/t5-small/"
tokenizer = AutoTokenizer.from_pretrained(underlying_model_name)
model = T5ForConditionalGeneration.from_pretrained(underlying_model_name)

