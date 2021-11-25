import pandas as pd
import ast
from tqdm import tqdm

df = pd.read_csv("v4_atomic_tst.csv")
out = "test_flat.tsv"

ii =0
data = []
pbar = tqdm(total = len(df))
for idx, row in df.iterrows():
    ii+=1
    pbar.update(1)
    if False: #ii > 3:
        break
    for col,val in row.items():
        if not col in ["event", "split", "prefix"]:
            items = ast.literal_eval(val)
            for item in items:
                r = {}
                r["input_text"] = row["event"]
                r["prefix"] = col
                #print("prefix:",col,"item:", item)
                r["target_text"] = item
                data.append(r)

    

print("len data:", len(data))
df2 = pd.DataFrame(data, columns = ["prefix","input_text","target_text"])
print(df2.head())
df2.to_csv(out, sep="\t")
