import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet as wn
stop_words = set(stopwords.words('english'))

df = pd.read_table("train.tsv")
from tqdm import tqdm

rows = []
intersect = lambda x: set(x['input_text'].split(' ')).intersection(x['target_text'].split(' '))

#df = df.loc[df.prefix == "xIntent",:]
df["target_text"] = df["target_text"].astype(str)
ii = 0
stop_words = list(stop_words)
stop_words.extend(["PersonX", "PersonY", "PersonZ"])
bar = tqdm()
for idx, row in df.iterrows():
    com = intersect(row)
    wordsList = [w for w in com if not w in stop_words]
    if wordsList:
        tags = []
        has_verb = False
        for w in wordsList:
            synsets = wn.synsets(w)
            if synsets:
                 pos = synsets[0].pos()
                 tags.append((w,pos))
                 if pos == "v":
                     has_verb = True
                     break
        if has_verb:
            continue
        #tags = nltk.pos_tag(wordsList)
        ii +=1 
        bar.update(1)
        rows.append(row)
        if False:
            print("{:<40} {:<20} -- {:<20}".format(row["input_text"], row["prefix"],
                row["target_text"]))
            print("-----", com)
            print("-----", tags)
            print("==================================================")

df2 = pd.DataFrame(data=rows)
df2.to_csv("mytrain.tsv", sep="\t", index=False)
