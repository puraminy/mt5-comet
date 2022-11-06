
#!/bin/sh

g1=""
g2=""
for i in "$@"; do
   case $i in
       *) g1="${g1} $i";;
       --*) g2="${g2} $i";;
   esac
done

home=/home/ahmad
alias runlite="python ${home}/mt5-comet/comet/train/train.py"
# wrap experiments
path=${PWD##*/}          
cp train.sh ..


#def train(model_id, experiment, qtemp, anstemp, extemp, method, val_method, train_samples, test_set, val_samples, test_samples, sample_samples, load_path, data_path, train_path, val_path, test_path, sample_path, overwrite, save_path, output_name, lang, pred_tresh, ignore_blanks,only_blanks, include, exclude, nli_group, learning_rate, do_eval, cont, wrap, prefix, frozen, freez_step, unfreez_step, cpu, load_prompt_path, verbose, cycle, batch_size, path, from_dir, is_flax, config,clear_logs, gen_param, print_log, training_round, epochs_num, per_record, is_even, start, prompt_length, prompt_pos, zero_shot, sampling, opt_type, samples_per_head, deep_log, trans, encoder_type, from_words,rel_filter, ex_type, last_data, save_df, merge_prompts, num_workers, scorers, train_start, no_save_model, gen_bs, shared_embs, no_confirm, follow_method, repeat, trial, fz_parts, pid, use_dif_templates, break_sent,sort, do_preproc, replace_blanks, loop, know, show_samples, ph_num, save_data, pre_prefix, skip, use_all_data):


#runlite run -exp $path -bc base  -sd -var method=unsup-wrap--rel_filter=x_rels#multi --train_samples=300  --test_samples=1000 --repeat=3 --follow_method=True --loop=True
#extra="--cpu=True"

# different models
#runlite run -exp $path -bc base -ov $1 -var model_id=t5-v1--method=unsup-nat--rel_filter=xIntent#xNeed--train_samples=20#50--epochs_num=1#2--repeat=1#2--pid=0#1--learning_rate=1e-4#1e-5 -test_samples=300 --loop=True $extra --skip=True --follow_method=True 
runlite run -cpu -exp $path -bc base -ov $g1 -var model_id=t5-small--method=unsup-wrap-nat--rel_filter=xIntent-xAttr--train_samples=200--epochs_num=2--repeat=4--temp_num=64--loop=True--test_samples=100--merge_prompts=lstm --follow_method=True --scorers="rouge-bert" --data_path=${home}/mt5-comet/comet/data/atomic2020 --do_valid=False --val_samples=10 --encoder_type=lstml --cycle=100 $g2 --seed=123 --batch_size=8  





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
