alias runlite="python /home/pouramini/mt5-comet/comet/train/train.py"
# wrap experiments
#runlite run -exp $1 -bc base  -sd -var method=unsup-wrap--rel_filter=x_rels#multi --train_samples=300  --test_samples=1000 --repeat=3 --follow_method=True --loop=True

# normal experiments
runlite run -exp $1 -bc base  -sd $2 -var method=unsup-nat#sup-nat--rel_filter=x_rels#multi --train_samples=30  --test_samples=300 --repeat=3 --loop=True 
