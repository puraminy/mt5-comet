

python train.py train t5-large -n 0 \
  -eps 1 -lang en-en -vp atomic/test.tsv -tp atomic/train.tsv  \
  -load /home/pouramini/pret -gp greedy@True \
  -save /home/pouramini/pret -ow unsup-wrap-tokens-full \
  -mt unsup-wrap-tokens -w -et lstm -ot adam -f -bs 30 #-fw rel_tokens \
		   

