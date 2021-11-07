import pandas as pd
import json

with open("results.json", "r") as f:
    data = json.dumps(f)

df = pd.DataFrame(data).T
df.fillna(0, inplace=True)
print(df)

