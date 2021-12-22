

python train.py train t5-large -n 0 \
  -eps 1 -lang en-en -vp atomic/val_all_rels.tsv -tp atomic/train_45k.tsv -even  \
  -load /home/pouramini/pret -gp greedy@True \
  -save /home/pouramini/pret -ow unsup-wrap-tokens-mp-test \
  -mt unsup-wrap-tokens -w -et lstm -ot adam -f -bs 30 -mp #-fw rel_tokens \
		   

