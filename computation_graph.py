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

class ComputationGraph:
    class lossdict(dict):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)

        def set_decode(self, mapping):
            self.action_mapping = mapping

        def __str__(self):
            parts = []
            for k, v in self.items():
                parts.append('{0}: {1} -> {2}'.format(k[0], self.action_mapping[k[1]], str(v[-1])))
            return '\n'.join(parts)

    def __init__(self, env):
        """
        Initialize the computation graph

        The aim of the computation graph is to learn
        R(s,a,s')
        """
        ## Environment consist of the grid world and categories
        self.env = env
        self.dtype = env.dtype
        ## Category to location map
        self.matmap = env.get_matmap()
        ## Transition matrix P
        self.mattrans = env.get_mattrans()
        ## Learning rate alpha
        self.learning_rate = 0.001
        ## Discount factor
        self.gamma = 0.90
        ## Boltzmann temperature for Q softmax
        self.beta = 10.0
        ## Horizon - number of timesteps to simulate into the future from the current point
        self.horizon = 50
        ## Softmax Function, along dimension 1
        self.softmax = torch.nn.Softmax(dim=1)
        ## Number of gradient updates to compute
        self.num_updates = 100
        
        #initial random guess on r
        #r starts as [n] categories. we will inflate this to [rows,cols,n]
        # to multiple with matmap and sum across category
        #r = np.random.rand(5)*2-1
        ## Reward function as a function of category
        ## NOTE: Initialize reward to a constant value because a policy can
        # be the optimal under any constant reward function. Therefore,
        # initialize  r to 0 everywhere
        r = np.zeros(env.ncategories)
        self.rk = torch.tensor(r, dtype=self.dtype, requires_grad=True)

        ## Losses are recorded state by state for actions
        self.recorded_losses = ComputationGraph.lossdict()
        self.recorded_losses.set_decode({0: 'U', 1: 'R', 2: 'D', 3: 'L', 4: 'S'})

        self.fmax_Q = self.__softmax_Q
        self.fmax_options = {
            "softmax": self.__softmax_Q,
            "max": self.__max_Q,
            "min": self.__min_Q,
            "mean": self.__mean_Q
        }

    def dump_hyperparameters(self):
        return {
            'alpha': self.learning_rate,
            'gamma': self.gamma,
            'beta':  self.beta,
            'horizon': self.horizon,
            'num_updates': self.num_updates}


    def __softmax_Q(self, inQ):
        pi = self.softmax(self.beta * inQ)
        next_Q = (inQ * pi).sum(dim=1)
        return next_Q
        
    def __max_Q(self, inQ):
        next_Q, _ = torch.max(inQ, dim=1)
        return next_Q

    def __min_Q(self, inQ):
        next_Q, _ = torch.min(inQ, dim=1)
        return next_Q
    
    def __mean_Q(self, inQ):
        next_Q = torch.mean(inQ, dim=1)
        return next_Q

    def set_fmax(self, o):
        if o not in self.fmax_options:
            raise ValueError(f"{o} not in {self.fmax_options}")
        else:
            self.fmax_Q = self.fmax_options[o]

    def set_hyperparameters(self, d):
        hypers = {'alpha': self.set_alpha,
            'gamma': self.set_gamma,
            'beta': self.set_beta,
            'horizon': self.set_horizon,
            'num_updates': self.set_num_updates}

        for k, v in hypers.items():
            if k in d:
                v(d[k])

    def set_gamma(self, g):
        self.gamma = g

    def set_alpha(self, a):
        self.alpha = a
    
    def set_beta(self, b):
        self.beta = b

    def set_horizon(self, h):
        self.horizon = h
    
    def set_num_updates(self, n):
        self.num_updates = n

    def set_reward_estimate(self, r):
        self.rk = torch.tensor(r, dtype=self.dtype, requires_grad=True)

    def current_reward_estimate(self):
        return copy.deepcopy(self.rk.cpu().detach().numpy())

    def forward(self, as_numpy=False):
        """
        This is the planning function. For the agent's estimation of the reward function,
        what are the estimated Q values for taking each action, and what is the corresponding stochastic policy?
        The stochastic policy is computed using the softmax over the Q values.
        """
        ## Expand the reward function to map onto the matrix map
        new_rk = self.rk.unsqueeze(0)
        new_rk = new_rk.unsqueeze(0)
        nrows, ncols, ncategories = self.matmap.shape
        new_rk = new_rk.expand(nrows, ncols, ncategories)
        ## Dot product to obtain the reward function applied to the matrix map
        rfk = torch.mul(self.matmap, new_rk)
        rfk = rfk.sum(axis=-1)
        rffk = rfk.view(nrows*ncols) ## 1D view of the 2D grid
        #initialize the value function to be the reward function, required_grad should be false
        v = rffk.detach().clone()

        ## NOTE: Does the concept of a goal exist in this case? Agent isn't aware of the goal state
        ##       and the environment doesn't explicitly specify the goal state
        ## NOTE: Do rollouts to a specified horizon
        for _ in range(self.horizon):
            #inflate v to be able to multiply mattrans (need to do this every iteration)
            v = v.unsqueeze(0)
            v = v.expand(nrows*ncols, nrows*ncols)
            ## Compute Q values
            ## This forces Q to be (nrows*ncols x nacts)
            Q = torch.mul(self.mattrans, v).sum(dim=-1).T
            next_Q = self.fmax_Q(Q)
            v = rffk + self.gamma * next_Q ## This back to 1D view again

        pi = self.softmax(self.beta * Q)
        if not as_numpy:
            return pi, Q, v
        else:
            return pi.cpu().detach().numpy(), Q.cpu().detach().numpy(), v.cpu().detach().numpy()

    def compute_expert_loss(self, pi, trajacts, trajcoords):
        nrows, ncols = self.env.world.shape
        loss = 0

        ## Update the losses for all the demonstrations that have been seen so far
        for sa in self.recorded_losses:
            state, acti = sa
            example_loss = -torch.log(pi[state[0]*ncols+state[1]][acti])
            self.recorded_losses[sa].append(example_loss.cpu().detach().numpy())

        ## Update the new loss if applicable
        for i in range(len(trajacts)):
            acti = trajacts[i]
            state = trajcoords[i]
            loss += -torch.log(pi[state[0]*ncols+state[1]][acti])

            if (state, acti) not in self.recorded_losses:
                self.recorded_losses[(state,acti)] = [loss.cpu().detach().numpy()]
        return loss

    def action_loss(self, pi, Q, trajacts, trajcoords):
        nrows, ncols = self.env.world.shape
        loss = 0
        for i in range(len(trajacts)):
            acti = trajacts[i]
            state = trajcoords[i]
            loss += -torch.log(pi[state[0]*ncols+state[1]][acti])
        ## Should be averaged over the batch size:
        loss /= len(trajacts)
        loss.backward()
        return loss

    def action_update(self, trajacts, trajcoords):
        ## return self.__action_heuristic_update(trajacts, trajcoords)
        return self.__action_finite_iterations_update(trajacts, trajcoords)

    def __action_finite_iterations_update(self, trajacts, trajcoords):
        print("LEARNING!***************")
        piout, Qout, loss = None, None, None
        for k in range(self.num_updates):
            piout, Qout, _ = self.forward()
            loss = self.action_loss(piout, Qout, trajacts, trajcoords)
            with torch.no_grad():
                grads_value = self.rk.grad
                self.rk -= self.learning_rate * grads_value
                self.rk.grad.zero_()
        return piout, Qout, loss, self.rk

    def __action_heuristic_update(self, trajacts, trajcoords):
        print("LEARNING!***************")
        nrows, ncols = self.env.world.shape
        acts = self.env.actions
        grid_map = self.env.world
        trans_dict = self.env.flattened_sas_transitions()
        trans_tuple = self.env.all_sas_transitions(trans_dict)

        current_R = self.current_reward_estimate()
        updated_R = self.current_reward_estimate()

        cr0 = GridWorldRewardModel(current_R, cg.env, cg.gamma, trans_dict, trans_tuple)
        cr1 = GridWorldRewardModel(updated_R, cg.env, cg.gamma, trans_dict, trans_tuple)

        piout, Qout, loss = None, None, None

        #for k in range(self.num_updates):
        n = 0
        while n < 1 or (cr0.epic_distance(cr1, trans_tuple).cpu().detach().numpy() < 0.1):
            piout, Qout, _ = self.forward()
            loss = self.action_loss(piout, Qout, trajacts, trajcoords)
            with torch.no_grad():
                grads_value = self.rk.grad
                self.rk -= self.learning_rate * grads_value
                self.rk.grad.zero_()
            updated_R =  self.current_reward_estimate()
            #cr1 = GridWorldRewardModel(updated_R, cg.env, cg.gamma, trans_dict, trans_tuple)
            n += 1
            if n > 5000:
                break
            cr1.update(updated_R)
        print(f"Update Iterations: {n}")
        return piout, Qout, loss, self.rk

    def scalar_loss(self, pi, Q, trajacts, trajcoords, scalar):
        nrows, ncols = self.env.world.shape
        loss = 0
        for i in range(len(trajacts)):
            acti = trajacts[i]
            state = trajcoords[i]
            loss += -torch.log(pi[state[0]*ncols+state[1]][acti])
        loss.backward()
        return loss

    def scalar_scale(self, pi, action, r,c, scalar):
        def clip(v, nmin, nmax):
            if v < nmin: v = nmin
            if v > nmax-1: v = nmax-1
            return(v)

        nrows, ncols = self.env.world.shape
        scale = clip(scalar / pi[r*ncols + c][action].item(), -1, 2)
        return scale

    def scalar_update(self, scalar, action, r, c):
        print("LEARNING!***************")
        piout, Qout, loss = None, None, None
        ## NOTE: Scalar feedback performs a SINGLE on-policy update
        for k in range(1):
            piout, Qout, _ = self.forward()
            loss = self.scalar_loss(piout, Qout, [action], [(r,c)], scalar)
            scale = self.scalar_scale(piout, action, r, c, scalar)
            with torch.no_grad():
                grads_value = self.rk.grad
                self.rk -= (self.learning_rate * grads_value) * scale
                self.rk.grad.zero_()
        return piout, Qout, loss, self.rk

    def plotpolicy(self, pi):
        def findpol(grid,pi,r,c, iteration):
            ## Commented out to allow agent to roam anywhere
            #if grid[r][c] != 6: return
            iteration += 1
            if grid[r][c] == 10: return
            maxprob = max(pi[r*ncols+c,:])
            a = 6
            for ana in range(5):
                if pi[r*ncols+c, ana] == maxprob: a = ana
            grid[r][c] = a
            r += self.env.actions[a][0]
            c += self.env.actions[a][1]
            if iteration < 989:
                findpol(grid,pi,r,c,iteration)
            else:
                print(f'Exceeded iterations')
                return

        startr, startc = self.env.viz_start[0], self.env.viz_start[1]
        nrows, ncols = self.env.world.shape
        grid = []
        iteration = 0
        for r in range(nrows):
            line = []
            for c in range(ncols):
                line += [self.env.world[r][c]+6]
            grid += [line]
        findpol(grid,pi,startr,startc, iteration)
        for r in range(nrows):
            line = ""
            for c in range(ncols):
                line += '^>v<x?012345678'[grid[r][c]]
            print(line)

    def chooseAction(self, r, c):
        pi, Q, _ = self.forward()
        epsilon = 0.25
        nrows, ncols, ncategories = self.matmap.shape
        action_prob = pi[r*ncols+c].cpu().detach().numpy()
        print("Original Action Probabilities (Up, Right, Down, Left, Stay): ")
        print(np.round(action_prob, 3))
        ## Filter out invalid actions
        if r == 0:
            action_prob[0] = 0
        if c == ncols - 1:
            action_prob[1] = 0
        if r == nrows - 1:
            action_prob[2] = 0
        if c == 0:
            action_prob[3] = 0

        ## Renormalize probabilities
        action_prob = action_prob / np.sum(action_prob)
        print("Action Probabilities (Up, Right, Down, Left, Stay): ")
        print(np.round(action_prob, 3))

        ## Epsilon-greedy (currently disabled)
        # r = random.uniform(0, 1)
        # print("Random Number: " + str(r))

        # if r < epsilon:
        #   permitable_actions = np.nonzero(action_prob)[0]
        #   choice = np.random.choice(permitable_actions, 1)[0]
        #   print("Picking a random action...")
        #   print(choice)
        #   return choice
        # print("Picking from probabilities...")
        # choice = np.random.choice(5, 1, p=action_prob)[0]
        # print(choice)
        choice = np.argmax(action_prob)
        return choice, copy.deepcopy(pi[r*ncols+c].cpu().detach().numpy())

    def request_feedback(self, cur_r, cur_c, force=True):
        """
        Uses heuristics to determine whether or not to request feedback
        from the human.

        Current heuristics: number of violations, goal success rate,
        "have I recieved feedback for this state?"

        Stochastic Policy values
        """
        if force:
            return force

        nrows, ncols = self.env.world.shape
        acts = self.env.actions
        grid_map = self.env.world

        def plan(rs, cs, pi, can_reach, steps, max_steps):
            r, c = rs, cs
            success = 0.0
            violations = 0
            while steps < max_steps:
                if grid_map[r][c] == 4 or can_reach[r*ncols + c] > 0.5:
                    success = 1.0
                    break
                maxprob = max(pi[r*ncols+c, :])
                a = 6
                for ana in range(5):
                    if pi[r*ncols+c, ana] == maxprob: a = ana
                r += acts[a][0]
                c += acts[a][1]
                steps += 1
                if grid_map[r][c] in (1,2,3):
                    violations += 1
            return success, violations

        def trajSuccess(rs, cs, pi):
            """
            Computes goal success of a trajectory
            """
            success = np.zeros(nrows*ncols)
            goal, violations = plan(rs, cs, pi, success, 0, nrows*ncols)
            return goal, violations

        def goalSuccess(pi):
            """
            For every possible state, does planning with
            the current policy lead to the goal state?
            """
            success = np.zeros(nrows*ncols)
            for i in range(nrows):
                for j in range(ncols):
                    success[i*ncols + j], _ = plan(i, j, pi, success, 0, nrows*ncols)
            return np.mean(success)

        def policyViolations(pi, do_print=True):
            violation_map = np.zeros((nrows, ncols))
            iteration_map = np.zeros((nrows, ncols))
            for i in range(nrows):
                for j in range(ncols):
                    grid = []
                    for r in range(nrows):
                        line = []
                        for c in range(ncols):
                            line += [6]
                        grid += [line]
                    it, viol = stateViolations(grid, pi, i, j)
                    violation_map[i][j] = viol
                    iteration_map[i][j] = it
            if do_print:
                print("Policy Violation Map:")
                print(violation_map)
                print("Iteration Map:")
                print(iteration_map)
                print("Average Policy Violation Count: " + str(np.mean(violation_map)))
                # print("Standard Deviation Violation Count: " + str(round(np.std(violation_map), 3)))
                print("Average Iteration Count: " + str(np.mean(iteration_map)))
                # print("Standard Deviation Iteration Count: " + str(round(np.std(iteration_map), 3)))
            return iteration_map, violation_map
            # returns number of violations in a state

        def stateViolations(grid, pi, r, c):
            if grid[r][c] != 6: return (0, 0)
            maxprob = max(pi[r*ncols+c, :])
            a = 6
            for ana in range(5):
                if pi[r*ncols+c, ana] == maxprob: a = ana
            grid[r][c] = a
            r += acts[a][0]
            c += acts[a][1]
            it, viol = stateViolations(grid, pi, r, c)
            if grid[r][c] < 4:
                it += 1
            tile_type = grid_map[r][c]
            if tile_type == 1 or tile_type == 2 or tile_type == 3:
                viol += 1
            if tile_type == 0 and a == 4:
                viol += 1 ## Violation by staying in the 0 zone
            return (it, viol)

        ## Compute Heuristics for making the decisions
        pi, Q, _ = cg.forward()
        success_rate = goalSuccess(pi)
        iters, viols = policyViolations(pi)
        #success_rate, viols = trajSuccess(cur_r, cur_c, pi)

        if success_rate < 1.0 or np.mean(viols) > 0.02:
            return True
        else:
            return False
