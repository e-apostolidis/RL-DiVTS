# -*- coding: utf-8 -*-
# Bash script to automate the procedure of evaluating an experiment.
# The script is taking as granted that the experiments are saved under "base_path" as "base_path/$EXP_ID/$DATASET_NAME".
# Run the evaluation scripts of the `exp_id` experiment for a specific `dataset_name`.
# First, get the training losses from tensorboard as a csv file, for each data-split.
# Then, compute the P@k (k=1, 3) based on the top-3 and top-1 selected user thumbnails, respectively.

# example of use: sh evaluate.sh exp1 OVP
base_path="../RL-DiVTS/Summaries"
exp_id=$1
dataset_name=$2

exp_path="$base_path/$exp_id/$dataset_name"

echo "Extract training data from log files for $exp_id of $dataset_name dataset ..."
for i in $(seq 0 1 9); do
	path="$exp_path/logs/split$i"
	python evaluation/exportTensorFlowLog.py "$path" "$path"
done

echo "Evaluation using the top-3 selected user thumbnails:"
python evaluation/compute_P@k.py "$exp_path" "$dataset_name"

echo "Evaluation using the top-1 selected user thumbnail:"
python evaluation/compute_P@k_on1thumb.py "$exp_path" "$dataset_name"
