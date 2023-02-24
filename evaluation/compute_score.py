# -*- coding: utf-8 -*-
import sys
from choose_best_epoch import choose_epoch
import json
import numpy as np
import h5py

ssim_threshold = 0.7  # Structural similarity threshold

# Process the input arguments
base_path = sys.argv[1]
init_id = int(sys.argv[2])
dataset = sys.argv[3]

top_k = 3

h5_file_path = f"../data/{dataset}/{dataset.lower()}.h5"

AVGAcc_final = []
for exp_id in range(init_id, init_id + 5):
    exp_path = f"{base_path}{exp_id}/{dataset}"
    best_epochs = choose_epoch(exp_path)
    print(f"\tSelected models (training epochs): \t\t {best_epochs}")

    hdf = h5py.File(h5_file_path, 'r')
    AVGAcc_exp = 0

    for split_id in range(5):
        AVGAcc_split = 0

        epoch = best_epochs[split_id]

        results_file = f"{exp_path}/results/split{split_id}/{dataset}_{epoch}.json"
        with open(results_file) as f:
            data = json.loads(f.read())
            video_names = list(data.keys())

        for video_name in video_names:
            imp_scores = np.asarray(data[video_name])         # Read the results (importance scores) for a video and
            sorted_score_inds = np.argsort(imp_scores)[::-1]  # sort the frames (desc.) according to the predicted scores

            # Read the initial number of frames for the video (before sampling)
            n_frames = np.array(hdf.get(f"{video_name}/n_frames"))
            interval = round(n_frames / imp_scores.shape[0])

            top3_indices = []
            for index in sorted_score_inds[:3]:
                top3_indices.append(int(index * interval))

            # Read the ssim matrix and the ground truth thumbnail-indices from the h5
            ssim_matrix = np.array(hdf.get(f"{video_name}/ssim_matrix"))
            top_thumbnail_ids = np.array(hdf.get(f"{video_name}/top{top_k}_thumbnail_ids"))

            # AVGAcc computation
            AVGAcc = 0
            for my_index in top3_indices[:3]:
                for gt_index in top_thumbnail_ids[:top_k]:
                    # Check if any of the top3 GT thumbnail is similar [relevant] to any of the selected ones
                    if ssim_matrix[my_index, gt_index] > ssim_threshold:
                        AVGAcc = AVGAcc + 1
                        break
            AVGAcc_split = AVGAcc_split + (AVGAcc / 3)

        # compute AVGAcc for each split
        AVGAcc_split = AVGAcc_split / len(video_names)

        AVGAcc_exp += AVGAcc_split
        # print(f"Performance (AVGAcc) on split {split_id}: {P3_split}")

    AVGAcc_final.append(AVGAcc_exp / 5)
    hdf.close()

AVGAcc_final = np.round(np.array(AVGAcc_final) * 100, 3)
print(f"\tAverage accuracy of top-3 matching over all splits: {AVGAcc_final.mean():.2f} \u00B1 {AVGAcc_final.std(ddof=1):.2f}.")
