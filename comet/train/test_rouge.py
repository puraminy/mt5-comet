from rouge import Rouge 

hyp = ["2", "3"]
refs = ["3 3", "2 2"]
rouge = Rouge()

score1 = rouge.get_scores(hyps=hyp, refs= refs, avg=True, ignore_empty=True)
#score2 = r2.calc_score(hyp, refs)
print(score1)
#print(score2)


