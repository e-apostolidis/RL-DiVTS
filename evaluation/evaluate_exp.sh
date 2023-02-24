# -*- coding: utf-8 -*-
# Bash script to automate the procedure of evaluating a single experiment.
# The script is taking as granted that the experiments are saved under "base_path" as "base_path/$EXP_ID/$DATASET_NAME".
# Run the evaluation scripts for the experiment for a specific `dataset_name`.
# First, get the training losses from tensorboard as a csv file, for each data-split.
# Then, compute the measures based on the $criterion and the top-3 and top-1 selected user thumbnails, respectively.

# example of use: sh evaluation/evaluate_exp.sh 10 OVP precision
base_path="../RL-DiVTS/Thumbnails/exp"
exp_id=$1
dataset_name=$2
criterion=$3

echo "Extract training data from log files for exp$exp_id of $dataset_name dataset ..."
for split_j in $(seq 0 1 9); do
	path="$base_path$exp_id/$dataset_name/logs/split$split_j"
  python evaluation/exportTensorFlowLog.py "$path" "$path"
done

echo "Evaluation using the top-3 selected user thumbnails:"
python evaluation/compute_score_single.py "$base_path" "$exp_id" "$dataset_name" 3 "$criterion"

echo "Evaluation using the top-1 selected user thumbnail:"
python evaluation/compute_score_single.py "$base_path" "$exp_id" "$dataset_name" 1 "$criterion"
