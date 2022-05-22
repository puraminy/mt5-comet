alias runlite="python train.py"
# wrap experiments
#runlite run -exp $1 -bc base  -sd -var method=unsup-wrap--rel_filter=x_rels#multi --train_samples=300  --test_samples=1000 --repeat=3 --follow_method=True --loop=True
#extra="--cpu=True"

# different models
runlite run -exp base-v1-lmb -bc base -ov -sm $1 -var model_id=t5-base#t5-v1#t5-lmb--method=unsup-nat#sup-nat#sup#unsup--rel_filter=xIntent--train_samples=100#1  --test_samples=300 --repeat=3 --loop=True $extra --skip=True 

# learning rate for supervised and unsupervised learning for t5-v1
#runlite run -exp learning-rate -bc base -ov -sm $1 --model_id=t5-v1 -var train_samples=100#200#300--learning_rate=0.0001#0.00001--method=sup#sup-nat#unsup-nat  --rel_filter=xIntent --train_samples=100  --test_samples=300 --repeat=3 --loop=True $extra --skip=True 


# learning rate for supervised and unsupervised learning for t5-v1
#runlite run -exp learning-rate -bc base -ov -sm $1 --model_id=t5-v1 -var method=unsup-nat#sup-nat--train_samples=100#200#300--loop=True#False  --rel_filter=xIntent  --test_samples=300 --repeat=3 $extra --skip=True 


#runlite run -exp samples_per_head -bc base  -sd $2 -var method=unsup-nat#sup-nat--samples_per_head=2#3 --train_samples=100  --test_samples=300 --repeat=3 --loop=True $extra 

# samples_per_head 
#runlite run -exp samples_per_head -bc base  -sd $2 -var method=unsup-nat#sup-nat--samples_per_head=2#3 --train_samples=100  --test_samples=300 --repeat=3 --loop=True $extra 

# repeatition with dif temps or not experiments
#runlite run -exp dif_temps -bc base  -sd $2 -var method=unsup-nat#sup-nat--use_dif_templates=True#False--rel_filter=xIntent#xAttr --train_samples=100  --test_samples=300 --repeat=3 --loop=True --samples_per_head=2 $extra 

# break sent experiments
#runlite run -exp break_sent -bc base  -sd $2 -var method=unsup-nat#sup-nat--break_sent=True#False --train_samples=100  --test_samples=300 -repeat=3 --loop=True --samples_per_head=2 $extra 
