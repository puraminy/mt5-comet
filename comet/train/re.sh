python train.py run \
	-exp lr-rate-sup \
	-bc base \
	-var method=sup#sup-end--learning_rate=0.001#0.0001#0.00001#0.000001#0.005#0.0005#0.00005#0.000005#0.0000005 \
	--train_samples=1800 \
       	--test_samples=1000 \
	--follow_method=True \
	--repeat=3 \
