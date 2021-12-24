import pandas as pd
import torch

class MyDataset(torch.utils.data.IterableDataset):
    def __init__(self, fname, until=10):
        self.df = pd.read_table("atomic/" + fname)
        self.until = until
    def preproc(self, t):
        pre = t[0]
        post = t[1]
        text = "Preproc: " + pre + "|" + post
        print(text)
        return text

    def __iter__(self):
        _iter = self.df_iter()
        return map(self.preproc, _iter)

    def df_iter(self):
        ret = []
        for idx, row in self.df.iterrows():
             ret.append((row["prefix"],row["input_text"]))
        return iter(ret)



