from sentence_transformers import CrossEncoder
model = CrossEncoder('/home/pouramini/pret/mm/nli-roberta-base')
list = [('A man is eating pizza', 'A man eats something'), ('A black race car starts up in front of a crowd of people.', 'A man is driving down a lonely road.')]
scores = model.predict(list)

#Convert scores to labels
label_mapping = ['contradiction', 'entailment', 'neutral']
labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
print("===================")
for p in list:
    s = model.predict(p)
    print(p)
    print(s)
    m  = s.argmax()
    l = label_mapping[m]
    print(l)

print("===================")
for p,s, l in zip(list,scores, labels):
    print(p)
    print(s)
    print(l)

