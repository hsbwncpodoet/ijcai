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

Transition = namedtuple("Transition", "t reward_estimate state action policy feedback_type feedback_value")

class TrialBuffer:
    def __init__(self):
        self.all_trials = []
        self.feedback_statistics = {}

    def register_trial(self, trial):
        self.all_trials.append(trial)
        for k,v in trial.feedback_indices.items():
            if k not in self.feedback_statistics:
                self.feedback_statistics[k] = []

    def update_statistics(self):
        trial = self.all_trials[-1]
        for feedback_type, indices in trial.feedback_indices.items():
            self.feedback_statistics[feedback_type].append(len(indices))

    def sample_feedback_type(self, feedback_type, batch_size):
        ## How many samples are in our buffer?
        total_current = len(self.all_trials[-1].feedback_indices[feedback_type])
        total_feedback = [x for x in self.feedback_statistics[feedback_type]]
        total_feedback.append(total_current)
        total = sum(total_feedback)
        selection = None
        if batch_size < total:
            selection = random.sample(range(total), batch_size)
        else:
            selection = [x for x in range(total)]

        ## Sort in the indices for iteration
        selection.sort()
        transitions = [None for x in range(len(selection))]
        trial_idx = 0
        selection_idx = 0
        prev_accum = 0
        for x in accumulate(total_feedback):
            while selection_idx < len(selection) and selection[selection_idx] < x:
                idx = selection[selection_idx] - prev_accum
                try:
                    data_idx = self.all_trials[trial_idx].feedback_indices[feedback_type][idx]
                except IndexError:
                    print(f'Selection: {selection}')
                    print(f'x: {x}, selection[selection_idx]={selection[selection_idx]}')
                    print(f'prev_accum: {prev_accum}, idx={idx}')
                    print(f'trial_idx: {trial_idx}')
                    print(f'total_feedback: {total_feedback}')
                transitions[selection_idx] = self.all_trials[trial_idx].transitions[data_idx]
                selection_idx += 1
            prev_accum = x
            trial_idx += 1
        return transitions

class Trial:
    def __init__(self, initial_state, goal, reward_estimate, grid_map):
        self.initial_state = initial_state
        self.goal = goal
        self.reward_estimate = reward_estimate
        self.grid_map = grid_map
        self.transitions = []
        self.feedback_indices = dict()

    def amount_of_feedback_value(self, key, val):
        count = 0
        if key not in self.feedback_indices:
            return 0
        else:
            for idx in self.feedback_indices[key]:
                if val == self.transitions[idx].feedback_value:
                    count += 1
            return count

    def amount_of(self, feedback_type):
        if feedback_type not in self.feedback_indices:
            return 0
        else:
            return len(self.feedback_indices[feedback_type])

    def get_effort(self, feedback_types):
        return sum(len(self.feedback_indices[k]) for k in feedback_types)

    def register_feedback_type(self, feedback_type):
        self.feedback_indices[feedback_type] = []

    def add_transition(self, t, r, s, a, p, ft, fv):
        self.transitions.append(Transition(t, r, s, a, p, ft, fv))
        if ft in self.feedback_indices:
            idx = len(self.transitions) - 1
            self.feedback_indices[ft].append(idx)

    def update_last_transition_reward(self, r):
        last_transition = self.transitions[-1]
        self.transitions[-1] = Transition(*((last_transition[0], r) + last_transition[2:]))
