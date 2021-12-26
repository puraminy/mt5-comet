from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast, AutoTokenizer 
    )
import click
from termcolor import colored

@click.command()
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
def main(path):
    print("Loading model...")
    underlying_model_name = path
    tokenizer = AutoTokenizer.from_pretrained(underlying_model_name)
    model = T5ForConditionalGeneration.from_pretrained(underlying_model_name)


    my_spacials = [x for x in tokenizer.additional_special_tokens if not "<extra_id"  in x]
    print(my_spacials)
    document = ""
    input_text = ""
    while document != "end":
        inp = input("input:") or "@PersonX drinks tea" 
        if inp.startswith("@"):
            input_text = inp.replace("@","")
        document = inp.replace("#",input_text).replace("!",document).replace("@","")
        print(colored(document, 'green'))
        # encode input context
        doc_tokens = tokenizer.tokenize(document)
        #print("doc_tokens:", doc_tokens)
        doc_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
        input_ids = tokenizer(document, return_tensors="pt").input_ids
        #print("doc_ids:", doc_ids)
        #print("input_ids:", input_ids)
        # generate 3 independent sequences using beam search decoding (5 beams)
        # with T5 encoder-decoder model conditioned on short news article.
        outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
        print("Generated:", colored(tokenizer.batch_decode(outputs, skip_special_tokens=True),'blue'))

main()
