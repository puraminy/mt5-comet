import json
from comet.tokenization_t5 import EncDecTokenizer
from .EncDecDatasets import EncDecDataset
import pandas as pd
import os
import pathlib

class CBDataset(EncDecDataset):
    def __init__(self, args, tokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CBDataset, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []
    
        with open(self.path, "r") as f:
            lines = f.readlines()

        self.ratio = 1
        rows = [] 
        for line in lines[:int(self.ratio * len(lines))]:
            _data = {}
            d = json.loads(line)

            label_map = {
                "entailment": "yes",
                "contradiction": "no",
                "neutral": "maybe",
            }
            _data["input_text"] = d["hypothesis"] + "{@@@}" + d["premise"]
            _data["prefix"] = "cb" 
            _data["target_text"] = label_map[d["label"]]
            rows.append(_data)
            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(d["hypothesis"]) + [self.extra_id_0] + self.tokenizer.encode(d["premise"])
            else:
                context =  self.tokenizer.encode(d["hypothesis"] + "?") + [self.extra_id_0] + self.tokenizer.encode(".") + self.tokenizer.encode(d["premise"])

            target = [0, self.extra_id_0] + self.tokenizer.encode(label_map[d["label"]])

            data.append({
                "idx": d["id"] if self.do_infer else self.idx,  
                "index": d["id"] if self.do_infer else self.idx,  
                "enc_input_ids": context,
                "dec_input_ids": target[:-1], # not including label id
                "label_ids": target[1:], # not including leadnig zero
                "query": d["hypothesis"] + "? <extra_id_0>." + d["premise"],
                "event": d["hypothesis"],
                "resp": d["label"],
                "rel": label_map,
                "rep": 1,
            })

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1


        df = pd.DataFrame(data=rows)
        df.to_csv(self.path.replace(".jsonl",".tsv"), index=False, sep="\t")
        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class CBDatasetUni(EncDecDataset):
    def __init__(self, args, tokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CBDatasetUni, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)

            label_map = {
                "contradiction": 188,
                "neutral": 279,
                "entailment": 254,
            }

            if self.prompt_config:
                choice_ids = [188, 5] + self.tokenizer.encode("no") + [5] + [279, 5] + self.tokenizer.encode("maybe") + [5] + [254, 5] + self.tokenizer.encode("yes") + [5]
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(d["hypothesis"]) + [58] + self.tokenizer.encode(d["premise"]) + [5] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.extra_id_0]
            else:
                pass

            target = [0, self.extra_id_0] + [label_map[d["label"]]]

            data.append({
                "idx": d["id"] if self.do_infer else self.idx,  
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
                "query": d["premise"],
                "event": d["hypothesis"]
            })

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len

