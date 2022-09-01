import torch
import os
from torch.utils.data import Dataset
import pickle
import math
from comet.utils.rank import print_rank_0, save_rank_0

class EncDecDataset(Dataset):
    def __init__(self, args, tokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        self.args = args
        self.tokenizer = tokenizer
        self.ratio = ratio
        self.path = path
        self.pad_id = tokenizer.pad_token_id
        self.add_target_post=add_target_post
        self.split = split
        self.do_infer = do_infer
        self.idx = 0
        self.extra_id_0 = tokenizer.convert_tokens_to_ids(["<extra_id_0>"])[0]
        self.extra_id_1 = tokenizer.convert_tokens_to_ids(["<extra_id_1>"])[0]
        self.prompt_config = prompt_config
        if cache_path is not None:
            cache_path = os.path.join(cache_path, "cache_{}_{}.pkl".format(path.replace("/", "_"), ratio))
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.data, self.max_enc_len, self.max_dec_len = pickle.load(f)
            else:
                self.data, self.max_enc_len, self.max_dec_len = self.process_data()
                with open(cache_path, "wb") as f:
                    pickle.dump((self.data, self.max_enc_len, self.max_dec_len), f)
        else:
            self.data, self.max_enc_len, self.max_dec_len = self.process_data()

        if num > 0:
            self.data = self.data[:num]

        # if prompt_config is not None:
        #     self.data, self.max_enc_len, self.max_dec_len = self.add_prompt_ids(self.data, self.max_enc_len, self.max_dec_len)

        if do_infer:
            total_eval_batch_size = 1 * args.eval_batch_size
            total_data_num = math.ceil(len(self.data) / total_eval_batch_size) * total_eval_batch_size
            while len(self.data) < total_data_num:
                tmp = self.data[0].copy()
                tmp["idx"] = -1
                self.data.append(tmp)

        print_str = "Path: {} | Ratio:{} | Max enc len: {} | Max dec len: {} | Data num: {}".format(path, ratio, self.max_enc_len, self.max_dec_len, len(self.data))
        print_rank_0(print_str)
        save_rank_0(args, print_str)

    def process_data(self):
        raise NotImplementedError

    def add_prompt_ids(self, data, max_enc_len, max_dec_len):
        enc_prompt_ids = [i for i in range(self.prompt_config["enc"]["prompt_len"])]
        dec_prompt_ids = [i for i in range(self.prompt_config["dec"]["prompt_len"])]
        pad_ids = [self.tokenizer.pad_id for _ in range(self.prompt_config["dec"]["prompt_len"])]

        for d in data:
            d["input_ids"] = enc_prompt_ids + d["input_ids"]
            d["decoder_input_ids"] = dec_prompt_ids + d["decoder_input_ids"]
            d["label_ids"] = pad_ids + d["label_ids"]

        max_enc_len += self.prompt_config["enc"]["prompt_len"]
        max_dec_len += self.prompt_config["dec"]["prompt_len"]

        return data, max_enc_len, max_dec_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(self, samples):
        bs = len(samples)
        model_data = {
            "input_ids": torch.ones(bs, self.max_enc_len, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_enc_len),
            "decoder_attention_mask": torch.zeros(bs, self.max_dec_len),
            #"cross_attention_mask": torch.zeros(bs, 1, self.max_dec_len, self.max_enc_len),
            "decoder_input_ids": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
            "labels": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
        }
        if not self.do_infer:
            no_model_data = {
                "idx": torch.zeros(bs, dtype=torch.long),
                "labels": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
                "loss_mask": torch.zeros(bs, self.max_dec_len)
            }
        else:
            no_model_data = {
                "idx": torch.zeros(bs, dtype=torch.long),
            }

        #breakpoint()
        for i, samp in enumerate(samples):
            enc_len, dec_len = len(samp["enc_input_ids"]), len(samp["dec_input_ids"])
            model_data["input_ids"][i][:enc_len] = torch.tensor(samp["enc_input_ids"], dtype=torch.long)
            model_data["decoder_input_ids"][i][:dec_len] = torch.tensor(samp["dec_input_ids"], dtype=torch.long)
            model_data["attention_mask"][i][:enc_len] = samp.get("enc_attention_mask", 1.0)
            model_data["decoder_attention_mask"][i][:dec_len] = samp.get("dec_attention_mask", 1.0)
            #model_data["cross_attention_mask"][i][0, :dec_len, :enc_len] = samp.get("enc_cross_attention_mask", 1.0)
            #model_data["labels"][i][:len(samp["label_ids"])] = torch.tensor(samp["label_ids"], dtype=torch.long)
            no_model_data["idx"][i] = samp["idx"]
            if not self.do_infer:
                no_model_data["labels"][i][:len(samp["label_ids"])] = torch.tensor(samp["label_ids"], dtype=torch.long)
                if self.prompt_config is not None:
                    no_model_data["loss_mask"][i][self.prompt_config["dec"]["prompt_len"]:len(samp["label_ids"])] = 1.0
                else:
                    no_model_data["loss_mask"][i][:len(samp["label_ids"])] = 1.0

        if self.args.fp16:
            model_data["attention_mask"] = model_data["attention_mask"].half()
            model_data["decoder_attention_mask"] = model_data["decoder_attention_mask"].half()
            model_data["cross_attention_mask"] = model_data["cross_attention_mask"].half()

        return model_data, no_model_data
