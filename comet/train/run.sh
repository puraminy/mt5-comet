

python train.py train t5-large -n 0 -even -exp t5-large-unsup-wrap-tokens-lstm-embed \
  -eps 1 -lang en-en -vp atomic/val_all_rels.tsv  \
  -load /home/pouramini/pret -gp greedy@True \
  -save /home/pouramini/pret -ow unsup-wrap-tokens -bs 4 \
  -mt unsup-wrap-tokens -w -et lstm -ot adam -f -bs 60 #-fw rel_tokens \
		   

