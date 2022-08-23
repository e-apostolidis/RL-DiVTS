# -*- coding: utf-8 -*-
import torch.nn as nn


class RL_DiVTS(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """ Scoring LSTM to assess the importance of each video frame.

        :param int input_size: The number of expected features in the input.
        :param int hidden_size: The number of features in the hidden state
        :param int num_layers: Number of recurrent layers.
        """
        super(RL_DiVTS, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(hidden_size * 2, 1),  # bidirectional => scalar
                                 nn.Sigmoid())

    def forward(self, features):
        """ Produces frames importance scores from the frame features, using a bidirectional LSTM.

        :param torch.Tensor features: Frame features with shape [T, 1, input_size]
        :return: A tensor with shape [T, 1] containing the frames' importance scores in [0, 1].
        """
        self.lstm.flatten_parameters()
        features, _ = self.lstm(features)       # [T, 1, hidden_size * 2]
        scores = self.out(features.squeeze(1))  # [T, 1]

        return scores


if __name__ == '__main__':
    pass
