# -*- coding: utf-8 -*-
import torch.optim as optim
from torch.autograd import Variable
from reward import aesthetics_reward, representativeness_reward
import json
import numpy as np
import random
from tqdm import tqdm, trange
import os
from torch.distributions import Categorical
import torch.nn.functional as F
from layers import RL_DiVTS
from utils import TensorboardWriter

weight_factor = {"OVP": 1000, "Youtube": 25000}


class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """ Class that Builds, Trains and Evaluates the RL-DiVTS model. """
        # Initialize variables to None, to be safe
        self.model, self.optimizer, self.writer = None, None, None

        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Set the seed for generating reproducible random numbers
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

    def build(self):
        """ Function for constructing the RL-DiVTS model, its key modules and parameters. """
        # Model creation
        self.model = RL_DiVTS(input_size=self.config.input_size,
                              hidden_size=self.config.hidden_size,
                              num_layers=self.config.num_layers).to(self.config.device)

        if self.config.mode == 'train':
            self.optimizer = optim.Adam(list(self.model.parameters()), lr=self.config.lr)
            self.writer = TensorboardWriter(str(self.config.log_dir))

    def train(self):
        """ Main function to train the RL-DiVTS model. """
        num_of_picks = self.config.selected_thumbs      # Number of selected thumbnails

        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            self.model.train()

            # Baseline rewards for videos
            train_keys = self.train_loader.dataset.split['train_keys']
            baselines = {key: 0. for key in train_keys}

            reward_history, rep_reward_history, aes_reward_history = [], [], []
            num_batches = int(len(self.train_loader) / self.config.batch_size)  # full-batch or mini batch
            iterator = iter(self.train_loader)
            for batch in range(num_batches):
                self.optimizer.zero_grad()

                if self.config.verbose:
                    tqdm.write('Training Scorer...')

                for video in range(self.config.batch_size):
                    # load the frame features and the pre-computed aesthetic scores of the current sample (video)
                    image_features, video_name, aes_scores_mean = next(iterator)

                    # get the frame-level aesthetic scores ready for use
                    aesthetic_quality = aes_scores_mean.squeeze(0).to(self.config.device)  # [T]

                    image_features = image_features.view(-1, self.config.input_size)       # [T, input_size]
                    image_features_ = Variable(image_features).to(self.config.device)
                    num_of_frames = image_features_.shape[0]

                    # compute the frame-level importance scores
                    original_features = image_features_.unsqueeze(1)  # [T, 1, input_size]
                    scores = self.model(original_features)            # [T, 1]
                    importance = scores.squeeze(1)                    # [T]

                    # compute the overall score for each frame
                    starting_overall_score = aesthetic_quality * importance

                    # apply min-max normalization
                    min_score = torch.min(starting_overall_score)
                    max_score = torch.max(starting_overall_score)
                    starting_overall_score = (starting_overall_score - min_score) / (max_score - min_score)

                    # Frame Picking Mechanism
                    sl = torch.tensor(0, dtype=torch.float32, device=self.config.device, requires_grad=True)
                    s_loss = sl.clone()

                    # compute the pair-wise frame similarity scores
                    x_unit = F.normalize(image_features_, p=2, dim=1)
                    similarity = x_unit @ x_unit.t()
                    similarity.fill_diagonal_(1.)

                    epis_rewards, epis_aes_rewards, epis_rep_rewards = [], [], []
                    for _ in range(self.config.n_episodes):
                        picks, log_prob_over_picks = [], []

                        overall_score = starting_overall_score.clone()
                        for pick in range(num_of_picks):
                            if torch.all(overall_score <= 0):
                                print("The overall score for each frame consists of all negative or zero values.")

                            dist = Categorical(overall_score)
                            picked_frame = dist.sample()     # returns a scalar between 0 and (T - 1)
                            picks.append(picked_frame)

                            log_prob_picks = dist.log_prob(picked_frame)
                            log_prob_over_picks.append(log_prob_picks)

                            overall_score = overall_score * (1 - similarity[picked_frame])

                            # re-apply min-max normalization
                            min_score = torch.min(overall_score)
                            max_score = torch.max(overall_score)
                            overall_score = (overall_score - min_score) / (max_score - min_score)

                        picks_binary = (torch.zeros(num_of_frames)).to(self.config.device)
                        picks_binary[torch.stack(picks)] = 1.

                        # Compute the Aesthetics Reward for the selected set of thumbnails
                        aes_reward = aesthetics_reward(aesthetic_quality, picks_binary, num_of_picks)

                        # Compute the Representativeness Reward for the selected set of thumbnails
                        rep_reward = weight_factor[self.config.video_type] * representativeness_reward(image_features_, picks)

                        # Compute the overall reward
                        reward = (0.5 * rep_reward) + (0.5 * aes_reward)

                        log_prob_over_episode = torch.mean(torch.stack(log_prob_over_picks))
                        expected_reward = log_prob_over_episode * (reward - baselines[video_name[0]])
                        s_loss -= expected_reward       # minimize negative expected reward

                        epis_rewards.append(torch.tensor([reward], dtype=torch.float, device=self.config.device))
                        epis_aes_rewards.append(torch.tensor([aes_reward], dtype=torch.float, device=self.config.device))
                        epis_rep_rewards.append(torch.tensor([rep_reward], dtype=torch.float, device=self.config.device))

                    s_loss.backward()
                    baselines[video_name[0]] = 0.9 * baselines[video_name[0]] + 0.1 * torch.mean(torch.stack(epis_rewards))

                    reward_mean = torch.mean(torch.stack(epis_rewards))
                    rep_reward_mean = torch.mean(torch.stack(epis_rep_rewards))
                    aes_reward_mean = torch.mean(torch.stack(epis_aes_rewards))

                    reward_history.append(reward_mean)
                    rep_reward_history.append(rep_reward_mean)
                    aes_reward_history.append(aes_reward_mean)

                # Update model parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

                torch.cuda.empty_cache()

            reward_epoch = torch.mean(torch.stack(reward_history))
            rep_reward_epoch = torch.mean(torch.stack(rep_reward_history))
            aes_reward_epoch = torch.mean(torch.stack(aes_reward_history))

            # Plot
            if self.config.verbose:
                tqdm.write('Plotting...')
            self.writer.update_loss(reward_epoch, epoch_i, 'reward_epoch')
            self.writer.update_loss(rep_reward_epoch, epoch_i, 'representativeness_reward_epoch')
            self.writer.update_loss(aes_reward_epoch, epoch_i, 'aesthetics_reward_epoch')

            # Uncomment to save parameters at checkpoint
            if not os.path.exists(self.config.save_dir):
                os.makedirs(self.config.save_dir)
            # ckpt_path = str(self.config.save_dir) + f'/epoch-{epoch_i}.pkl'
            # if self.config.verbose:
            #    tqdm.write(f'Save parameters at {ckpt_path}')
            # torch.save(self.model.state_dict(), ckpt_path)

            self.evaluate(epoch_i)

    def evaluate(self, epoch_i):
        """ Saves the frame's importance scores for the test videos in json format.

        :param int epoch_i: The current training epoch.
        """
        self.model.eval()

        num_of_picks = self.config.selected_thumbs
        out_dict = {}
        for image_features, video_name, aes_scores_mean in tqdm(self.test_loader, desc='Evaluate', ncols=80, leave=False):
            image_features = image_features.view(-1, self.config.input_size)           # [T, input_size]
            image_features_ = Variable(image_features).to(self.config.device)
            original_features = image_features_.unsqueeze(1)                           # [T, 1, input_size]
            num_of_frames = image_features_.shape[0]

            with torch.no_grad():
                # Get the frame-level aesthetic scores ready for use
                aesthetic_quality = aes_scores_mean.squeeze(0).to(self.config.device)  # [T]

                # Compute the frame-level importance scores
                scores = self.model(original_features)                                 # [T, 1]
                importance = scores.squeeze(1)                                         # [T]

                # Compute the overall score for each frame
                overall_score = aesthetic_quality * importance

                # Apply min-max normalization
                min_score = torch.min(overall_score)
                max_score = torch.max(overall_score)
                overall_score = (overall_score - min_score) / (max_score - min_score)

                # Compute the pair-wise frame similarity scores
                x_unit = F.normalize(image_features_, p=2, dim=1)
                similarity = x_unit @ x_unit.t()
                similarity.fill_diagonal_(1.)

                # Frame Picking Mechanism
                picks = []
                for pick in range(num_of_picks):
                    dist = Categorical(overall_score)
                    picked_frame = dist.sample()        # returns a scalar between 0 and (T - 1)
                    picks.append(picked_frame)

                    overall_score = overall_score * (1 - similarity[picked_frame])

                    # Re-apply min-max normalization
                    min_score = torch.min(overall_score)
                    max_score = torch.max(overall_score)
                    overall_score = (overall_score - min_score) / (max_score - min_score)

                picks_binary = (torch.zeros(num_of_frames)).to(self.config.device)
                picks_binary[torch.stack(picks)] = 1.

                increase_picks = (torch.ones(num_of_frames)).to(self.config.device)
                increase_picks = increase_picks + picks_binary

                weighted_scores = increase_picks.unsqueeze(1) * scores        # [T, 1]
                weighted_scores = weighted_scores.squeeze(1)
                weighted_scores = weighted_scores.cpu().numpy().tolist()

                out_dict[video_name] = weighted_scores

            if not os.path.exists(self.config.score_dir):
                os.makedirs(self.config.score_dir)

            score_save_path = self.config.score_dir.joinpath(f'{self.config.video_type}_{epoch_i}.json')
            with open(score_save_path, 'w') as f:
                if self.config.verbose:
                    tqdm.write(f'Saving score at {str(score_save_path)}.')
                json.dump(out_dict, f)
            score_save_path.chmod(0o777)


if __name__ == '__main__':
    pass
