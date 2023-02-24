# -*- coding: utf-8 -*-
# Bash script to automate the training of the model for the `Youtube` dataset.
# Runs main.py 5 times, for each data-split and for the selected number of epochs and batch size.

for exp_id in $(seq 1 5); do
  for split_id in $(seq 0 4); do
    python model/main.py --split_index "$split_id" --n_epochs 150 --batch_size 32 --video_type 'Youtube' --exp "$exp_id"
  done
done
