
#!/bin/sh


g1=""
g2=""
for i in $@
do
   case $i in
       # -- option
       --*) g1="${g1} $i"; g=1;;
       
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
          else
            others="$others $p"
          fi
      ;;
   esac
done
home=$(echo $others | xargs)
if [ -z $home ]; then
   home=/home/ahmad
fi 
echo $home
alias runlite="python ${home}/mt5-comet/comet/train/train.py"
# wrap experiments
folder=${PWD##*/}          
cp train.sh ..

test=100
train=200
#test=-1
#train=2
exp=xint-ex-want
log=${home}/logs/${exp}

runlite run -exp $exp -lp ${log} -bc base -ov $g2 -var method=unsup-wrap-nat--rel_filter=xIntent--train_samples=$train--epochs_num=2--repeat=4--temp_num=642#652--loop=True--test_samples=$test--merge_prompts=none--shared_embs=False --follow_method=True --scorers="rouge-bert" --data_path=${home}/mt5-comet/comet/data/atomic2020 --do_valid=False --val_samples=10 --encoder_type=lstm --cycle=100 $g1 --seed=123 --batch_size=16 

if [ $home = "/content" ]; then
	tar -czvf ${home}/${exp}.tar.gz ${log}
	cp ${home}/${exp}.tar.gz ${home}/drive/MyDrive/logs2 
fi

# learning rate for supervised and unsupervised learning for t5-v1
#runlite run -exp learning-rate -bc base -ov -sm $1 --model_id=t5-v1 -var train_samples=100#200#300--learning_rate=0.0001#0.00001--method=sup#sup-nat#unsup-nat  --rel_filter=xIntent --train_samples=100  --test_samples=300 --repeat=3 --loop=True $extra --skip=True 


