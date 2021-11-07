import pandas as pd
import json

with open("results.json", "r") as f:
    data = json.load(f)
results = {}
def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield (key, True)
            yield from recursive_items(value)
        else:
            yield (key, value)


for key, value in recursive_items(data):
    print(key, value)


