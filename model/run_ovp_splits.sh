# -*- coding: utf-8 -*-
# Bash script to automate the training of the model for the `OVP` dataset.
# Runs main.py 10 times, for each data-split and for the selected number of epochs and batch size.

for exp_id in $(seq 10 19); do
  for split_id in $(seq 0 9); do
    python model/main.py --split_index "$split_id" --n_epochs 200 --batch_size 40 --video_type 'OVP' --exp "$exp_id"
  done
done
