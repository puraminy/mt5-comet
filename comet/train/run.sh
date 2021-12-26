

python train.py train t5-base -n 0 \
  -eps 1 -lang en-en -vp atomic/test.tsv -tp atomic/train_45k.tsv -tstart 1000 \
  -load /home/pouramini/pret -gp greedy@True \
  -save /home/pouramini/pret -ow unsup-wrap-tokens-start-full_BASE_test_$1 \
  -mt unsup-tokens-start -bs 4 #-w -et lstm -ot adam -f -bs 10 -ts sample #-fw rel_tokens \
		   

