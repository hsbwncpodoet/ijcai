import random
from itertools import accumulate
import copy
import pickle
import datetime
import argparse
import os.path
from collections import namedtuple
from numpy.random import choice
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class GridWorldRewardModel:
    def __init__(self, reward_features, env, gamma, trans_dict, trans_tuple):
        self.gamma = gamma
        self.env = env
        self.matmap = self.env.get_matmap()
        nrows, ncols, ncategories = self.matmap.shape
        self.nrows = nrows
        self.ncols = ncols
        self.ncats = ncategories
        ## Grid map is a 2D representation of the GridWorld. Each element is the category
        self.grid_map = torch.tensor(self.env.world, dtype=int, requires_grad=False)
        ## Observation and action space is given by the flattened grid_world, it is 1D.
        self.obs_space = torch.arange(nrows*ncols, dtype=int)
        self.act_space = torch.arange(len(self.env.actions), dtype=int)
        ## Represents possible actions
        self.actions = torch.arange(len(self.env.actions), dtype=int, requires_grad=False)
        ## R(s,a,s') = R(s,a,\phi(s')), feature based rewards vector
        self.trans_dict = trans_dict
        self.trans_tuple = trans_tuple

        self.feature_rewards = torch.tensor(reward_features, dtype=self.env.dtype, requires_grad=True)
        self._forward()

    def _forward(self):
        new_rk = self.feature_rewards.unsqueeze(0)
        new_rk = new_rk.unsqueeze(0)
        new_rk = new_rk.expand(self.nrows, self.ncols, self.ncats)
        ## Dot product to obtain the reward function applied to the matrix map
        rfk = torch.mul(self.matmap, new_rk)
        rfk = rfk.sum(axis=-1) ## 2D representation, i.e. recieve R((r,c)) reward for arriving at (r,c)
        self.reward_model = rfk.view(self.nrows*self.ncols) ## flattened 1D view of the 2D grid
        ## Create 3D verasion of rewrd model: (s,a,s'). The above version corresponds with s'
        ## R(s,a,s')
        self.full_reward_model = torch.zeros((self.nrows*self.ncols, len(self.env.actions), self.nrows*self.ncols))
        for s,a,sp in self.trans_tuple:
            self.full_reward_model[s,a,sp] = self.reward_model[sp]
        self.canonicalized_reward = self.get_canonicalized_reward(self.trans_dict, self.trans_tuple)

    def update(self, reward_features):
        self.feature_rewards = torch.tensor(reward_features, dtype=self.env.dtype, requires_grad=True)
        self._forward()

    def expected_reward_from_s(self, s, transitions):
        """
        Computes the mean reward exactly starting in state s, averaged
        over the possible (a,s') allowed from state s
        (s,a,s') are in terms of 1D representation, i.e. s, s' \in {0,...,|S|-1}

        transitions is a {s: {a: s'}} dict
        """
        return torch.mean(self.reward_model[[sp for a, sp in transitions[s].items()]])

    def expected_reward_over_sas(self, transitions):
        """
        transitions is a tuple( tuple(s,a,s'), ... )
        """
        ## NOTE: We need to make a 1D tensor that gathers from specific indices represented by sasp
        ## This can be done with the 1D reward model
        return torch.mean(self.reward_model[[sasp[2] for sasp in transitions]])

    def get_canonicalized_reward(self, transitions_dict, transitions_tuple):
        canonicalized = torch.clone(self.full_reward_model) ## R(s,a,s')
        ## Below, used to compute R(s',A,S') and R(s,A,S')
        mean_from_state = torch.tensor(
            [self.expected_reward_from_s(state, transitions_dict) for state in range(self.env.world.size)]
        )

        ## Compute E[R(S,A,S')]
        mean_reward = torch.sum(self.full_reward_model)/len(transitions_tuple)
        for s,a,sp in transitions_tuple:
            canonicalized[s,a,sp] += (self.gamma*mean_from_state[sp] - mean_from_state[s] - self.gamma*mean_reward)
        return canonicalized

    def epic_distance(self, other, samples):
        """
        sample is (s,a,s') tuples
        """
        shape = self.canonicalized_reward.shape
        S = shape[0]
        A = shape[1]
        ra = torch.flatten(self.canonicalized_reward)
        rb = torch.flatten(other.canonicalized_reward)

        idx = (lambda s,a,sp: s*A*S + a*S + sp)
        ## Sample indices
        indices = np.array([idx(s,a,sp) for s,a,sp in samples])
        return self.pearson_distance(ra[indices], rb[indices])

    def pearson_distance(self, ra, rb):
        mu_a = torch.mean(ra)
        mu_b = torch.mean(rb)
        var_a = torch.mean(torch.square(ra - mu_a))
        var_b = torch.mean(torch.square(rb - mu_b))

        cov = torch.mean((ra - mu_a) * (rb - mu_b))
        corr = cov / torch.sqrt(var_a * var_b)
        corr = torch.clamp(corr, -1.0, 1.0)
        return torch.sqrt((1.0-corr)/2.0)
