# -*- coding: utf-8 -*-
import sys
from choose_best_epoch import choose_epoch
import json
import numpy as np
import h5py

permitted_top_k = [1, 3]
permitted_criterions = ["matching", "precision"]
ssim_threshold = 0.7  # Structural similarity threshold

# Process the input arguments
base_path = sys.argv[1]
init_id = int(sys.argv[2])
dataset = sys.argv[3]
assert int(sys.argv[4]) in permitted_top_k, f"The number of ground truth thumbnails must be : {*permitted_top_k,}"
assert sys.argv[5] in permitted_criterions, f"The criterion must be : {*permitted_criterions,}"
top_k = int(sys.argv[4])
criterion = sys.argv[5]

h5_file_path = f"../RL-DiVTS/data/{dataset}/{dataset.lower()}.h5"
P1_final, P3_final = [], []
for exp_id in range(init_id, init_id + 10):
    exp_path = f"{base_path}{exp_id}/{dataset}"
    best_epochs = choose_epoch(exp_path)
    # print(f"\tSelected models (training epochs): \t\t {best_epochs}")

    hdf = h5py.File(h5_file_path, 'r')
    P1_exp, P3_exp = 0, 0
    for split_id in range(10):
        P1_split, P3_split = 0, 0

        epoch = best_epochs[split_id]
        results_file = f"{exp_path}/results/split{split_id}/{dataset}_{epoch}.json"
        with open(results_file) as f:
            data = json.loads(f.read())
            video_names = list(data.keys())

        for video_name in video_names:
            imp_scores = np.asarray(data[video_name])         # Read the results (importance scores) for a video and
            sorted_score_inds = np.argsort(imp_scores)[::-1]  # sort the frames (desc.) according to the imp. score

            # Read the initial number of frames for the video (before sampling)
            n_frames = np.array(hdf.get(f"{video_name}/n_frames"))
            interval = round(n_frames / imp_scores.shape[0])

            top3_indices = []
            for index in sorted_score_inds[:3]:
                top3_indices.append(int(index * interval))

            # Read the ssim matrix and the ground truth thumbnail-indices from the h5
            ssim_matrix = np.array(hdf.get(f"{video_name}/ssim_matrix"))
            top_thumbnail_ids = np.array(hdf.get(f"{video_name}/top{top_k}_thumbnail_ids"))

            # No draws during the matching criterion
            if criterion == "matching":
                top_thumbnail_ids = top_thumbnail_ids[:top_k]

            # P@1 computation
            my_index = top3_indices[0]  # the top 1 thumbnail
            P1 = 0
            for gt_index in top_thumbnail_ids:
                # Check if any of the top3 GT thumbnail is similar [relevant] to the selected one
                if ssim_matrix[my_index, gt_index] > ssim_threshold:
                    P1 = 1
                    break
            P1_split += P1

            # P@3 computation
            P3 = 0
            for my_index in top3_indices[:3]:
                for gt_index in top_thumbnail_ids:
                    # Check if any of the top3 GT thumbnail is similar [relevant] to any of the selected ones
                    if ssim_matrix[my_index, gt_index] > ssim_threshold:
                        P3 = P3 + 1
                        break
            if criterion == "precision":
                P3_split = P3_split + min(P3, 1)
            else:
                P3_split = P3_split + (P3 / 3)

        # Find the P1 and P3 for each split
        P1_split = P1_split / len(video_names)
        P3_split = P3_split / len(video_names)

        P1_exp += P1_split
        P3_exp += P3_split
    # print(f"Performance (P@1 & P@3) on split {split}: {P1_split}, {P3_split}")

    P1_final.append(P1_exp / 10)  # number of splits
    P3_final.append(P3_exp / 10)
    hdf.close()

P1_final = np.round(np.array(P1_final) * 100, 3)
P3_final = np.round(np.array(P3_final) * 100, 3)
if criterion == "precision":
    print(
        f"\tAverage P@1 performance over all splits: {P1_final.mean():.2f} \u00B1 {P1_final.std(ddof=1):.2f}.")
    print(
        f"\tAverage P@3 performance over all splits: {P3_final.mean():.2f} \u00B1 {P3_final.std(ddof=1):.2f}.")
elif criterion == "matching" and top_k == 3:
    print(
        f"\tAverage top-3 matching over all splits: {P3_final.mean():.2f} \u00B1 {P3_final.std(ddof=1):.2f}.")
else:
    print(f"\tUnsupported criterion and top-k thumbnails combination!")
