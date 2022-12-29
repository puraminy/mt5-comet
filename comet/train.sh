
#!/bin/sh

g1=""
g2=""
for i in $@
do
   case $i in
       # -- option
       --*) g1="${g1} $i"; g=1;;
       
       -m) echo "------"; g=3;;
       # - option
       -*) g2="${g2} $i"; g=2;;
       
       # Parameter 
       *) p=$i
          if [ "$g" = 1 ]
          then
            g1="${g1} $p"
            g=0
          elif [ "$g" = 2 ]
          then
            g2="${g2} $p"
            g=0
          elif [ "$g" = 3 ]
          then
            m=$p 
            g=0
          else
            others="$others $p"
          fi
      ;;
   esac
done
home=$(echo $others | xargs)
model=t5-base
case "$HOME" in 
  *ahmad*)
    # Do stuff
    model=t5-base
    ;;
esac
if [ -z $home ]; then
   home=$HOME
fi 
echo $home
alias runlite="python3 ${home}/mt5-comet/comet/train/train.py"
# wrap experiments
folder=${PWD##*/}          

test=100
train=200
if [ -z $m ]; then
   m=11
fi
echo "m: ${m}"
if [ "$m" -eq "0" ]; then
  echo "testing train"
  test=-1
  train=2
elif [ "$m" -eq "1" ]; then
  echo "testing train and test"
  test=8
  train=8
fi
seed=123

exp=xint-att-cc-xx2
log=${home}/logs   #/${exp}
echo "log: ${log}"
#filter=xIntent#xAttr#xNeed#xReact#xEffect#oReact#xWant#multi
filter=xIntent #xAttr#multi
merge=none #lstm
tn=com-pre #com-mid #merge-mid-nat #com-mid-nat #mat #2#4#5
shared=False
trial=2
epochs=3
# tag=temp_num@encoder_type@trunc_router@prompt_token_num
trunc=none #sign #sigmoid#sign
enc_type=lstm
router=fixed #learned
fex=none #skills_weight#none
skill=none #learned#private#none
mt=unsup-wrap-nat
cfg=configs/baselines/finetuning.json

runlite run -exp $exp -cfg ${cfg} -lp ${log} -bc base -ov $g2 -var method=${mt}--rel_filter=$filter--train_samples=$train--epochs_num=$epochs--prompt_token_num=8--repeat=4--temp_num=$tn--loop=custom--test_samples=$test--flat_prompts=$merge--frozen=False--seed=123--n_prompts=1--trunc_router=$trunc--router_lr=0.001--learning_rate=0.5--encoder_type=$enc_type--router_variant=$router--freeze_exclude=$fex--stype=t5--skilled_variant=$skill--n_skills=4--follow_method=True--arg-prefix_dim=30#200 --scorers="rouge-bert" --data_path=${home}/mt5-comet/comet/data/atomic2020/sel --do_valid=False --val_samples=10  --cycle=100 $g1 --batch_size=8 --trial=$trial  --model_id=$model --do_valid=True 

# configs/baselines/prompt_tuning.json 
# configs/attempt/single_task.json  
#--freeze_parts="router" --freeze_step=500 

#--unfreeze_parts="encoder" --unfreez_step=50 

#cp train.sh ${log}
case "$home" in 
  *content*)
    # Do stuff
	mv /content/*time*.log ${log}/$exp
	tar -czvf /content/${exp}-$m.tar.gz ${log}/$exp
	cp /content/${exp}-$m.tar.gz ${home}/logs 
    ;;
esac

# learning rate for supervised and unsupervised learning for t5-v1
#runlite run -exp learning-rate -bc base -ov -sm $1 --model_id=t5-v1 -var train_samples=100#200#300--learning_rate=0.0001#0.00001--method=sup#sup-nat#unsup-nat  --rel_filter=xIntent --train_samples=100  --test_samples=300 --repeat=3 --loop=True $extra --skip=True 


