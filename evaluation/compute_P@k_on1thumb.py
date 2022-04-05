# -*- coding: utf-8 -*-
import sys
from choose_best_epoch import choose_epoch
import json
import numpy as np
import h5py

ssim_threshold = 0.7  	# Structural similarity threshold

exp_path = sys.argv[1]
dataset = sys.argv[2]
h5_file_path = '../RL-DiVTS/data/' + dataset + '/' + dataset.lower() + '.h5'

best_epochs = choose_epoch(exp_path)
print(f"\tSelected models (training epochs): \t\t {best_epochs}")

hdf = h5py.File(h5_file_path, 'r')

P1_final, P3_final = 0, 0
for split in range(10):
	P1_split, P3_split = 0, 0

	epoch = best_epochs[split]
	results_file = exp_path + '/results/split' + str(split) + '/' + dataset + '_' + str(epoch) + '.json'
	with open(results_file) as f:
		data = json.loads(f.read())
		video_names = list(data.keys())

	for video_name in video_names:
		imp_scores = np.asarray(data[video_name])  # Read the results (importance scores) for a video and
		sorted_score_inds = np.argsort(imp_scores)[::-1]  # sort them (desc.) to find the frames with the max imp. score

		# Read the initial number of frames for the video (before sampling)
		n_frames = np.array(hdf.get(video_name+'/n_frames'))
		interval = round(n_frames / imp_scores.shape[0])

		top3_indices = []
		for index in sorted_score_inds[:3]:
			top3_indices.append(int(index * interval))

		# Read the ssim matrix and the ground truth thumbnail-indices from the h5
		ssim_matrix = np.array(hdf.get(video_name+'/ssim_matrix'))
		top1_thumbnail_ids = np.array(hdf.get(video_name+'/top1_thumbnail_ids'))

		# P@1 computation
		my_index = top3_indices[0]  # the top 1 thumbnail
		P1 = 0
		for gt_index in top1_thumbnail_ids:
			if ssim_matrix[my_index, gt_index] > ssim_threshold:
				P1 = 1
				break
		P1_split += P1

		# P@3 computation
		P3 = 0
		for my_index in top3_indices[:3]:
			for gt_index in top1_thumbnail_ids:
				if ssim_matrix[my_index, gt_index] > ssim_threshold:
					P3 = 1
					break
			if P3 == 1:
				break
		P3_split += P3

	# Find the P1 and P3 for each split
	P1_split = P1_split / len(video_names)
	P3_split = P3_split / len(video_names)

	P1_final += P1_split
	P3_final += P3_split
	# print(f"Performance (P@1 & P@3) on split {split}: {P1_split}, {P3_split}")

P1_final = P1_final / 10
P3_final = P3_final / 10
print(f"\tAverage performance (P@1 & P@3) over all splits: {P1_final*100:.2f}, {P3_final*100:.2f}.")

hdf.close()
