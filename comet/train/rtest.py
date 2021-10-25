
from rouge_metric import PerlRouge

rouge = PerlRouge(rouge_n_max=3, rouge_l=True, rouge_w=True,
    rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)

# Load summary results and evaluate
hypotheses = [
    'how are you\ni am fine',                       # document 1: hypothesis
    'it is fine today\nwe won the football game',   # document 2: hypothesis
]
references = [[
    'how do you do\nfine thanks',   # document 1: reference 1
    'how old are you\ni am three',  # document 1: reference 2
], [
    'it is sunny today\nlet us go for a walk',  # document 2: reference 1
    'it is a terrible day\nwe lost the game',   # document 2: reference 2
]]

scores = rouge.evaluate(hypotheses, references)
print(scores)

