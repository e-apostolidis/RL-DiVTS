# -*- coding: utf-8 -*-
import csv
import torch


def choose_epoch(path):
    """ Finds the epoch with the maximum received overall reward, for each split.

    :param str path: The path to .csv file with the logged training data (received rewards).
    :return: A List[int] (0-based) that represents the selected epoch for each split.
    """

    best_epochs = []
    for split in range(0, 5):
        logs_file = path + '/logs/split' + str(split) + '/scalars.csv'
        losses = {}
        losses_names = []

        # Read the csv file with the logged training data (received rewards)
        with open(logs_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for (i, row) in enumerate(csv_reader):
                if i == 0:
                    for col in range(len(row)):
                        losses[row[col]] = []
                        losses_names.append(row[col])
                else:
                    for col in range(len(row)):
                        losses[losses_names[col]].append(float(row[col]))

        # criterion: received reward
        reward = losses['reward_epoch']
        reward_t = torch.tensor(reward)

        # normalize values
        reward_t = reward_t / max(reward_t)

        # keep the epoch that maximizes the received reward
        epoch = torch.argmax(reward_t)
        best_epochs.append(epoch.item())

    return best_epochs
