alias runlite="python /home/pouramini/mt5-comet/comet/train/train.py"
# wrap experiments
#runlite run -exp $1 -bc base  -sd -var method=unsup-wrap--rel_filter=x_rels#multi --train_samples=300  --test_samples=1000 --repeat=3 --follow_method=True --loop=True
extra="--cpu=True"
# samples_per_head 
runlite run -exp samples_per_head -bc base  -sd $2 -var method=unsup-nat#sup-nat--samples_per_head=2#3 --train_samples=100  --test_samples=300 --repeat=3 --loop=True $extra 

# repeatition with dif temps or not experiments
runlite run dif_temps $1 -bc base  -sd $2 -var method=unsup-nat#sup-nat--use_dif_templates=True#False--rel_filter=xIntent#xAttr --train_samples=100  --test_samples=300 --repeat=3 --loop=True --samples_per_head=2 $extra 

# break sent experiments
runlite run -exp break_sent -bc base  -sd $2 -var method=unsup-nat#sup-nat--break_sent=True#False --train_samples=100  --test_samples=300 -repeat=3 --loop=True --samples_per_head=2 $extra 
