import numpy as np
from comet.tokenization_t5 import EncDecTokenizer, extra_id_0, extra_id_1
import torch
from .IndexDataset import MMapIndexedDataset, _build_index_mappings


class CLSPretrainDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer: EncDecTokenizer, name, data_prefix, document_ids_in_splits, context_indexed_dataset: MMapIndexedDataset,
                target_indexed_dataset: MMapIndexedDataset, num_samples, enc_seq_length, dec_seq_length, prompt_config, seed):

        self.name = name
        self.context_indexed_dataset = context_indexed_dataset
        self.target_indexed_dataset = target_indexed_dataset
        self.tokenizer = tokenizer
        self.enc_seq_length = enc_seq_length
        self.dec_seq_length = dec_seq_length
        self.prompt_config = prompt_config
        self.enc_prompt_len = 0
        self.dec_prompt_len = 0
        if self.prompt_config is not None:
            self.enc_prompt_len = self.prompt_config["enc"]["prompt_len"]
            self.dec_prompt_len = self.prompt_config["dec"]["prompt_len"]

        # Checks
        assert np.min(document_ids_in_splits) >= 0
        assert np.max(document_ids_in_splits) < context_indexed_dataset.sizes.shape[0]

        # Build index mappings.
        self.doc_idx = _build_index_mappings(
            self.name, data_prefix, document_ids_in_splits, self.context_indexed_dataset.sizes,
            num_samples, enc_seq_length - 1, seed)
            # NOTE: enc_seq_length - 1: This function is originally designed for autoregressive models, so the output length is actually input length +1

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.doc_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        # NOTE: We do not get shuffle idx because the documents are already shuffled

        idx = self.doc_idx[idx]

        contexts = self.context_indexed_dataset.get(idx)
        targets = self.target_indexed_dataset.get(idx)

        contexts = [int(x) for x in contexts]
        end = contexts[-1:]
        contexts = contexts[:-1]
        contexts = contexts[:366] + end

        contexts = [-(i + 1) for i in range(int(self.enc_prompt_len))] + contexts

        targets = [int(x) for x in targets]

        labels = targets[1:]
        targets = targets[:-1]

        contexts = contexts + [self.tokenizer.pad_id] * (self.enc_seq_length - len(contexts))
        targets = targets + [self.tokenizer.pad_id] * (self.dec_seq_length - len(targets))
        labels = labels + [self.tokenizer.pad_id] * (self.dec_seq_length - len(labels))

        contexts = contexts[:self.enc_seq_length]
        targets = targets[:self.dec_seq_length]

        return {
            "contexts": np.array(contexts), 
            "targets": np.array(targets), 
            "labels": np.array(labels),
        }
