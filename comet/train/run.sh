python train.py train test -eval -n 0 \
  -start 0 -lang en-en -vp atomic/val_all_rels.tsv -ng 0 \
  -eps 1 -pl 5 -ppos end \
  -ow  temp -bs 4 \
  -mt unsup-wrap -w -et emb -ot adam -f -bs 32 #-fw rel \
