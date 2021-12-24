from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast, AutoTokenizer 
    )
underlying_model_name = "/home/ahmad/pret/t5-small/"
tokenizer = AutoTokenizer.from_pretrained(underlying_model_name)
model = T5ForConditionalGeneration.from_pretrained(underlying_model_name)


document = (
 "at least two people were killed in a suspected bomb attack on a passenger bus "
 "in the strife-torn southern philippines on monday , the military said."
)
# encode input context
input_ids = tokenizer(document, return_tensors="pt").input_ids
# generate 3 independent sequences using beam search decoding (5 beams)
# with T5 encoder-decoder model conditioned on short news article.
outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

