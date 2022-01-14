import random
from itertools import accumulate
import copy
import pickle
import datetime
import argparse
import os.path
from collections import namedtuple
from env.environment import Environment
from env.world import Worlds
from numpy.random import choice
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from computation_graph import ComputationGraph
from reward_model import GridWorldRewardModel
from feedback_buffer import Transition, TrialBuffer, Trial


def train(episodes, cg, prepop_trial_data=None, force_feedback=True, source_is_human=True, feedback_policy_type="human", mixed_strat=None, mixed_percent=0.5):
    """
    Train the agent's estimate of the reward function
    """
    tensor_time = []
    pi, Q, loss, rk = None, None, None, None
    max_steps = 100
    trial_buffer = TrialBuffer()
    ## Prepopulate the TrialBuffer with feedback examples
    if prepop_trial_data is not None:
        trial_buffer.register_trial(prepop_trial_data)
        trial_buffer.update_statistics()
        
    for ep in range(episodes):
        ## Initialize episode:
        ## Choose a random start state
        r, c = next(cg.env.random_start_state())
        g = cg.env.get_goal_state()

        trial_data = Trial((r,c), g, cg.current_reward_estimate(), cg.env.get_world())
        trial_data.register_feedback_type(cg.env.SCALAR_FEEDBACK)
        trial_data.register_feedback_type(cg.env.ACTION_FEEDBACK)
        trial_data.register_feedback_type(cg.env.NO_FEEDBACK)
        trial_buffer.register_trial(trial_data)

        steps = 0
        while cg.env.world[r,c] != g and steps < max_steps:
            ## Agent Plans an action given the current state, based on current policy
            action, local_policy = cg.chooseAction(r, c)

            if cg.request_feedback(r, c, force=force_feedback):

                ## Tell the human what action the agent plans to take from the current state:
                print(f"Agent is currently at ({r},{c}) on {cg.env.world[r,c]}")
                cg.env.visualize_environment(r,c)
                cg.env.inform_human(action)

                ## Get feedback for the planned action before actually taking it.
                ##feedback_str = cg.env.acquire_feedback(action, r, c, source_is_human=True, feedback_policy_type="human")
                feedback_str = cg.env.acquire_feedback(action, r, c, source_is_human=source_is_human, feedback_policy_type=feedback_policy_type, agent_cg=cg, mixed_strat=mixed_strat, mixed_percent=mixed_percent)
                feedback = None
                ## Classify the feedback
                feedback_type = cg.env.classify_feedback(feedback_str)
                if feedback_type == cg.env.SCALAR_FEEDBACK:
                    print("Scalar Feedback Provided")
                    scalar = cg.env.feedback_to_scalar(feedback_str)
                    feedback = scalar
                    pi, Q, loss, rk = cg.scalar_update(scalar, action, r, c)
                    print(f"Updated reward: {rk}")

                elif feedback_type == cg.env.ACTION_FEEDBACK:
                    print("Action Feedback Provided")
                    #Collect Trajectories
                    trajacts, trajcoords = cg.env.feedback_to_demonstration(feedback_str, r, c)
                    feedback = trajacts[0]
                    # trajacts, trajcoords = cg.env.acquire_human_demonstration(max_length=1)
                else:
                    print("Invalid feedback provided. Ignoring.")

                trial_data.add_transition(
                    steps, cg.current_reward_estimate(),
                    (r,c), action, local_policy, feedback_type, feedback
                )

                if feedback_type == cg.env.ACTION_FEEDBACK:
                    ## Update the reward function based on a batch of expert transitions from
                    ## the feedback buffer
                    batch_transitions = trial_buffer.sample_feedback_type(cg.env.ACTION_FEEDBACK, 100)
                    ## Convert transitions to trajacts, trajcoords
                    trajacts = [tr.feedback_value for tr in batch_transitions]
                    trajcoords = [tr.state for tr in batch_transitions]
                    pi, Q, loss, rk = cg.action_update(trajacts, trajcoords)

                    ## Because the reward in the transition should be from AFTER feedback
                    trial_data.update_last_transition_reward(cg.current_reward_estimate())

                    cg.compute_expert_loss(pi, [feedback], [(r,c)])
                    print(f"Updated reward: {rk}")
                print(cg.recorded_losses)
            else:
                trial_data.add_transition(
                    steps, cg.current_reward_estimate(),
                    (r,c), action, local_policy, cg.env.NO_FEEDBACK, None
                )

            ## endif cg.request_feedback()

            ## Perform actual transition
            r,c = cg.env.step(r,c, action)
            steps += 1

        if steps == max_steps:
            print(f"Agent unable to learn after {steps} steps, going to next trial.")
        trial_buffer.update_statistics()

        tensor_time.append(copy.deepcopy(rk))
        if ep % 1 == 0:
            for x in tensor_time:
                print(x)
            print('Plotting Policy:')
            cg.plotpolicy(pi)

    ## Save the data for this experiment. The environment is small enough
    ## that we can just regenerate the policy at each time step from the reward function.
    ## Note that the reward function is the reward AFTER feedback is obtained
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"exp_{ts}.pkl","wb") as f:
        pickle.dump(trial_buffer.all_trials, f)
        pickle.dump(cg.recorded_losses, f)

    return trial_buffer.all_trials, cg.recorded_losses

def compute_effort(data, cg):
    ## Effort = How much feedback was given
    effort = [trial.get_effort((cg.env.ACTION_FEEDBACK, cg.env.SCALAR_FEEDBACK)) for trial in data]
    ## Distribution: Quantity of Action, Quantity of Scalar
    dist = {
        cg.env.ACTION_FEEDBACK: None,
        cg.env.SCALAR_FEEDBACK: None,
        cg.env.NO_FEEDBACK: None
    }
    for k in dist:
        dist[k] = [trial.amount_of(k) for trial in data]

    ## Special case: scalar feedback of 0 should count as NO_FEEDBACK, so shift those counts from
    ## SCALAR_FEEDBACK to NO_FEEDBACK
    zero_feedback = [trial.amount_of_feedback_value(cg.env.SCALAR_FEEDBACK, 0) for trial in data]
    for idx in range(len(zero_feedback)):
        amt = zero_feedback[idx]
        dist[cg.env.SCALAR_FEEDBACK][idx] -= amt
        dist[cg.env.NO_FEEDBACK][idx] += amt
        effort[idx] -= amt

    return effort, dist

def extract_rewards(data, cg, last_only=False):
    """
    Computes and saves the rewards from each episode
    """
    as_numpy = True
    rewards = []
    trial_idx = 0
    for trial in data:
        ## Full metric calculations for every single timestep in the trial
        if not last_only:
            r = trial.reward_estimate
            print(f'Reawrd Estimate: {r}')
            rewards.append(r)
            for transition in trial.transitions:
                r = transition.reward_estimate
                print(f'Reawrd Estimate: {r}')
                rewards.append(r)
        ## Full metric calculation only for the last timestep in the trial
        else:
            if trial_idx == 0:
                r = trial.transitions[0].reward_estimate
                print(f'Reawrd Estimate: {r}')
                rewards.append(r)
                trial_idx += 1
    
            r = trial.transitions[-1].reward_estimate
            print(f'Reawrd Estimate: {r}')
            rewards.append(r)

    return rewards

def extract_policies(data, cg, last_only=False):
    """
    Computes and saves the policies from each episode
    """
    as_numpy = True
    policies = []
    trial_idx = 0
    for trial in data:
        ## Full metric calculations for every single timestep in the trial
        if not last_only:
            r = trial.reward_estimate
            print(f'Reawrd Estimate: {r}')
            cg.env.world = trial.grid_map
            cg.set_reward_estimate(r)
            pi, Q, _ = cg.forward(as_numpy)
            policies.append(pi)
            for transition in trial.transitions:
                r = transition.reward_estimate
                print(f'Reawrd Estimate: {r}')
                cg.set_reward_estimate(r)
                pi, Q, _ = cg.forward(as_numpy)
                policies.append(pi)
        ## Full metric calculation only for the last timestep in the trial
        else:
            if trial_idx == 0:
                r = trial.transitions[0].reward_estimate
                print(f'Reawrd Estimate: {r}')
                cg.env.world = trial.grid_map
                cg.set_reward_estimate(r)
                pi, Q, _ = cg.forward(as_numpy)
                policies.append(pi)
                trial_idx += 1
    
            r = trial.transitions[-1].reward_estimate
            print(f'Reawrd Estimate: {r}')
            cg.env.world = trial.grid_map
            cg.set_reward_estimate(r)
            pi, Q, _ = cg.forward(as_numpy)
            policies.append(pi)

    return policies

def stoch_goal_success(policies, cg):
    """
    Simulates the environment Markov Chain for N-steps, with the
    goal states as absorbing states.

    """
    nrows, ncols = cg.env.world.shape
    state_size = nrows*ncols
    arr_stochastic_goal_success = np.zeros(len(policies))
    N = state_size
    idx = 0
    for policy in policies:
        distribution, goal_indices = cg.env.simulate_markov_chain(policy, N)
        goal_success = 0.0
        for g in goal_indices:
            goal_success = goal_success + distribution[g]
        arr_stochastic_goal_success[idx] = goal_success
        print(f"Goal Success: {goal_success}")
        idx = idx + 1
    return arr_stochastic_goal_success


def stoch_policy_violations(policies, cg, violation_set):
    """
    Computes average 1-step policy violations

    Each policy is [|S| x |A|]. The policies are element-wise multiplied
    with the violations matrix
    For each state, a violating action is considered violating if
    
    1) The agent takes an action take allows it to end up in a violating state
    2) The agent takes the "stay" action
    """
    nrows, ncols = cg.env.world.shape
    state_size = nrows*ncols
    V = cg.env.get_violation_matrix(violation_set)
    arr_stochastic_violations = np.zeros(len(policies))
    idx = 0
    for policy in policies:
        ## Take the element-wise product, then add up all the probability mass
        ## and then divide by the state_size
        exp_violations = np.sum(V*policy) / state_size
        arr_stochastic_violations[idx] = exp_violations
        idx = idx + 1
    return arr_stochastic_violations

def save_policy_metrics(policies, cg, which="both"):
    nrows, ncols = cg.env.world.shape
    acts = cg.env.actions
    grid_map = cg.env.world

    ## Helpers for goal success
    def transition(r, c, a):
        r_next = r + acts[a][0]
        c_next = c + acts[a][1]
        r_next = min(r_next, nrows-1)
        r_next = max(r_next, 0)
        c_next = min(c_next, ncols-1)
        c_next = max(c_next, 0)
        return r_next, c_next
    def plan(rs, cs, pi, can_reach, steps, max_steps):
        """
        This does greedy planning
        """
        r, c = rs, cs
        success = 0
        while steps < max_steps:
            if grid_map[r][c] == 4 or can_reach[r*ncols + c] > 0.5:
                success = 1
                break
            a = np.argmax(pi[r*ncols+c])
            r, c = transition(r, c, a)
            steps += 1
        return success

    def goalSuccess(pi):
        """
        Deterministic goal success, with a greedy policy
        For every possible state, does planning with
        the current policy lead to the goal state?
        """
        success = np.zeros(nrows*ncols)
        for i in range(nrows):
            for j in range(ncols):
                success[i*ncols + j] = plan(i, j, pi, success, 0, nrows*ncols)
        return np.mean(success)
    
    ## Helpers for violations
    def policyViolations(pi, do_print=True):
        """
        Perform deterministic policy violations on rollouts:
        Starting in every state, do greedy policy rollouts until:
            - all states are visited, or
            - a loop is formed (a previous state is revisited)
        """
        violation_map = np.zeros((nrows, ncols))
        iteration_map = np.zeros((nrows, ncols))
        for i in range(nrows):
            for j in range(ncols):
                grid = np.ones((nrows,ncols), dtype=int)
                grid *= 6
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
        """
        Compute violations under a greedy deterministic policy
        """
        if grid[r][c] != 6: return (0, 0)
        a = np.argmax(pi[r*ncols+c])
        #maxprob = max(pi[r*ncols+c, :])
        #a = 6
        ## NOTE: Under below setup, the last max will be chosen.
        #for ana in range(5):
        #    if pi[r*ncols+c, ana] == maxprob: a = ana
        grid[r][c] = a
        #r += acts[a][0]
        #c += acts[a][1]
        r, c = transition(r, c, a)
        it, viol = stateViolations(grid, pi, r, c)
        if grid[r][c] < 4:
            it += 1
        tile_type = grid_map[r][c]
        if tile_type == 1 or tile_type == 2 or tile_type == 3:
            viol += 1
        if tile_type == 0 and a == 4:
            viol += 1 ## Violation by staying in the 0 zone
        return (it, viol)

    if which == "both":
        arr_goal_rates = np.zeros(len(policies))
        arr_violations = np.zeros(len(policies))

        idx = 0
        for policy in policies:
            ## Goal success
            success_rate = goalSuccess(policy)
            arr_goal_rates[idx] = success_rate
            ## Violations
            iteration_map, violation_map = policyViolations(policy)
            arr_violations[idx] = np.mean(violation_map)
            idx += 1
        return arr_goal_rates, arr_violations

    if which == "dpv":
        arr_violations = np.zeros(len(policies))
        idx = 0
        for policy in policies:
            ## Violations
            iteration_map, violation_map = policyViolations(policy)
            arr_violations[idx] = np.mean(violation_map)
            idx += 1
        return arr_violations

    if which == "dgs":
        arr_goal_rates = np.zeros(len(policies))
        idx = 0
        for policy in policies:
            ## Goal success
            success_rate = goalSuccess(policy)
            arr_goal_rates[idx] = success_rate
            idx += 1
        return arr_goal_rates

def compute_policy_eval_metrics(data, cg, last_only=False):
    """
    Computes policy goal success as defined in `compute_goal_success`
    Computes policy violations as defined in `compute_violations`

    We combine all the evaluations here so that we don't have to do planning
    more than once per time step
    """
    nrows, ncols = cg.env.world.shape
    acts = cg.env.actions
    grid_map = cg.env.world

    ## Helpers for goal success
    def plan(rs, cs, pi, can_reach, steps, max_steps):
        r, c = rs, cs
        success = 0
        while steps < max_steps:
            if grid_map[r][c] == 4 or can_reach[r*ncols + c] > 0.5:
                success = 1
                break
            maxprob = max(pi[r*ncols+c, :])
            a = 6
            for ana in range(5):
                if pi[r*ncols+c, ana] == maxprob: a = ana
            r += acts[a][0]
            c += acts[a][1]
            steps += 1
        return success

    def goalSuccess(pi):
        """
        For every possible state, does planning with
        the current policy lead to the goal state?
        """
        success = np.zeros(nrows*ncols)
        for i in range(nrows):
            for j in range(ncols):
                success[i*ncols + j] = plan(i, j, pi, success, 0, nrows*ncols)
        return np.mean(success)
    
    ## Helpers for violations
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
    
    goal_rate_per_trial = []
    violations_per_trial = []
    detailed_violations_per_trial = []
    for trial in data:
        goal_rate = []
        violations = []

        ## Full metric calculations for every single timestep in the trial
        if not last_only:

            r = trial.reward_estimate
            print(f'Reawrd Estimate: {r}')
            cg.env.world = trial.grid_map
            cg.set_reward_estimate(r)
            pi, Q, _ = cg.forward()

            ## Goal success
            success_rate = goalSuccess(pi)
            goal_rate.append(success_rate)
            ## Violations
            iteration_map, violation_map = policyViolations(pi)
            violations.append((np.mean(iteration_map),np.mean(violation_map)))
            detailed_violations_per_trial.append((iteration_map, violation_map))

            for transition in trial.transitions:
                r = transition.reward_estimate
                print(f'Reawrd Estimate: {r}')
                cg.set_reward_estimate(r)
                pi, Q, _ = cg.forward()

                ## Goal success
                success_rate = goalSuccess(pi)
                goal_rate.append(success_rate)
                ## Violations
                iteration_map, violation_map = policyViolations(pi)
                violations.append((np.mean(iteration_map),np.mean(violation_map)))
                detailed_violations_per_trial.append((iteration_map, violation_map))
        ## Full metric calculation only for the last timestep in the trial
        else:
            trial_length = len(trial.transitions)
            r = trial.transitions[trial_length - 1].reward_estimate
            print(f'Reawrd Estimate: {r}')
            cg.env.world = trial.grid_map
            cg.set_reward_estimate(r)
            pi, Q, _ = cg.forward()

            ## Goal success
            success_rate = goalSuccess(pi)
            goal_rate.append(success_rate)
            ## Violations
            iteration_map, violation_map = policyViolations(pi)
            violations.append((np.mean(iteration_map),np.mean(violation_map)))
            detailed_violations_per_trial.append((iteration_map, violation_map))

        goal_rate_per_trial.append(goal_rate)
        violations_per_trial.append(violations)
    return violations_per_trial, detailed_violations_per_trial, goal_rate_per_trial

def compute_goal_success(data, cg):
    """
    Computes the goal success at each timestep
    Computes the goal success at the end of the episode (= goal success of last timestep)
    """
    nrows, ncols = cg.env.world.shape
    acts = cg.env.actions
    grid_map = cg.env.world

    def plan(rs, cs, pi, can_reach, steps, max_steps):
        r, c = rs, cs
        success = 0
        while steps < max_steps:
            if grid_map[r][c] == 4 or can_reach[r*ncols + c] > 0.5:
                success = 1
                break
            maxprob = max(pi[r*ncols+c, :])
            a = 6
            for ana in range(5):
                if pi[r*ncols+c, ana] == maxprob: a = ana
            r += acts[a][0]
            c += acts[a][1]
            steps += 1
        return success

    def goalSuccess(pi):
        """
        For every possible state, does planning with
        the current policy lead to the goal state?
        """
        success = np.zeros(nrows*ncols)
        for i in range(nrows):
            for j in range(ncols):
                success[i*ncols + j] = plan(i, j, pi, success, 0, nrows*ncols)
        return np.mean(success)

    goal_rate_per_trial = []
    for trial in data:
        goal_rate = []
        r = trial.reward_estimate
        print(f'Reawrd Estimate: {r}')
        cg.env.world = trial.grid_map
        cg.set_reward_estimate(r)
        pi, Q, _ = cg.forward()
        success_rate = goalSuccess(pi)
        goal_rate.append(success_rate)
        for transition in trial.transitions:
            r = transition.reward_estimate
            print(f'Reawrd Estimate: {r}')
            cg.set_reward_estimate(r)
            pi, Q, _ = cg.forward()
            success_rate = goalSuccess(pi)
            goal_rate.append(success_rate)
        goal_rate_per_trial.append(goal_rate)
    return goal_rate_per_trial

def compute_violations(data, cg):
    """
    Computes the violations at the end of each episode
    Computes the violations during training of each timestep during each episode
    """
    nrows, ncols = cg.env.world.shape
    acts = cg.env.actions
    grid_map = cg.env.world

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

    violations_per_trial = []
    detailed_violations_per_trial = []
    for trial in data:
        violations = []
        r = trial.reward_estimate
        print(f'Reawrd Estimate: {r}')
        cg.env.world = trial.grid_map
        cg.set_reward_estimate(r)
        pi, Q, _ = cg.forward()
        iteration_map, violation_map = policyViolations(pi)
        violations.append((np.mean(iteration_map),np.mean(violation_map)))
        detailed_violations_per_trial.append((iteration_map, violation_map))
        for transition in trial.transitions:
            r = transition.reward_estimate
            print(f'Reawrd Estimate: {r}')
            cg.set_reward_estimate(r)
            pi, Q, _ = cg.forward()
            iteration_map, violation_map = policyViolations(pi)
            violations.append((np.mean(iteration_map),np.mean(violation_map)))
            detailed_violations_per_trial.append((iteration_map, violation_map))
        violations_per_trial.append(violations)

    return violations_per_trial, detailed_violations_per_trial

def min_mean_max_violations(violations):
    """
    Obtain the min, mean, and max of the input violations
    """
    min_violations = []
    max_violations = []
    avg_violations = []
    end_violations = []
    ini_violations = []
    len_trial = []
    for trial_violations in violations:
        min_violations.append(min(v[1] for v in trial_violations))
        max_violations.append(max(v[1] for v in trial_violations))
        end_violations.append(trial_violations[-1][1])
        ini_violations.append(trial_violations[0][1])
        steps = len(trial_violations)
        avg_violations.append(sum(v[1] for v in  trial_violations) / steps)
        len_trial.append(steps)

    return {
        'min': min_violations,
        'max': max_violations,
        'avg': avg_violations,
        'end': end_violations,
        'ini': ini_violations,
        'len': len_trial
    }

def min_mean_max_goal_success(rates):
    """
    Obtain the min, mean, and max of the input goal success rates
    """
    min_success = []
    max_success = []
    avg_success = []
    end_success = []
    ini_success = []
    len_trial = []
    for trial_rates in rates:
        min_success.append(min(r for r in trial_rates))
        max_success.append(max(r for r in trial_rates))
        end_success.append(trial_rates[-1])
        ini_success.append(trial_rates[0])
        steps = len(trial_rates)
        avg_success.append(sum(r for r in  trial_rates) / steps)
        len_trial.append(steps)

    return {
        'min': min_success,
        'max': max_success,
        'avg': avg_success,
        'end': end_success,
        'ini': ini_success,
        'len': len_trial
    }

def plot_group_violations(
    group_violations, group_success, group_effort,
    group_act, group_sca, group_non,
    group_pra, group_prs, group_prn, show=True):
    """
    Plots violations as a function of trial. Specifically,
    plots the max, mean, and min of a trail, averaged over a set of seeds.

    The structure of group_violations is:
    {
        grp_name: {
            'min': [average of seeds of ep1 val, ...],
            'max': [average of seeds of ep1 val, ...],
            'avg': [average of seeds of ep1 val, ...],
            'end': [average of seeds of ep1 val, ...],
        }
    }

    """
    styles = {
        'min': ':',
        'max': '--',
        'avg': '-.',
        'end': '-'
    }
    #color_list = ['blue','green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    #color_list = ['magenta','gold', 'green', 'blue', 'red', 'cyan', 'black']
    color_list = ['C0','C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
    colors = {k:v for k,v in zip(group_violations, color_list)}

    print(group_violations)

    ## Violation Averages
    for grp, avgs in group_violations.items():
        x = [n+1 for n in range(len(avgs['end']))]
        for k in ('min', 'max', 'end'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_'+k)
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Violations')
    plt.title('Violation Averages')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("violation_averages.per_trial.pdf", bbox_inches='tight')
        plt.close()

    ## Goal Success Averages
    for grp, avgs in group_success.items():
        x = [n+1 for n in range(len(avgs['end']))]
        for k in ('min', 'max', 'end'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_'+k)
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Goal Success')
    plt.title('Average Goal Success Rate')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("goal_success.per_trial.pdf", bbox_inches='tight')
        plt.close()

    ## Cumulative Effort
    for grp, avgs in group_effort.items():
        x = [n+1 for n in range(len(avgs['min']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, np.cumsum(avgs[k]), linestyle=styles[k], color=colors[grp], label=grp+'_effort')
        plt.fill_between(x, np.cumsum(avgs['min']), np.cumsum(avgs['max']), color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Effort')
    plt.title('Cumulative Effort')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("cumulative_effort.per_trial.pdf", bbox_inches='tight')
        plt.close()

    ## Length per Trial
    for grp, avgs in group_violations.items():
        x = [n+1 for n in range(len(avgs['len']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, avgs['len'], color=colors[grp], label=grp+'_len')
        
    ## Effort per Trial
    for grp, avgs in group_effort.items():
        x = [n+1 for n in range(len(avgs['min']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_effort')
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Effort')
    plt.title('Effort')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("effort.per_trial.pdf", bbox_inches='tight')
        plt.close()
    
    ## Feedback Types per Trial
    for grp, avgs in group_act.items():
        x = [n+1 for n in range(len(avgs['min']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_effort')
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Num Action')
    plt.title('Quantity Action Feedback')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("actions.per_trial.pdf", bbox_inches='tight')
        plt.close()
    ## Feedback Types per Trial
    for grp, avgs in group_pra.items():
        x = [n+1 for n in range(len(avgs['min']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_effort')
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Pecent Action')
    plt.title('Percent Action Feedback')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("percent_actions.per_trial.pdf", bbox_inches='tight')
        plt.close()

    ## Feedback Types per Trial
    for grp, avgs in group_sca.items():
        x = [n+1 for n in range(len(avgs['min']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_effort')
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Num Scalar')
    plt.title('Quantity Scalar Feedback')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("scalars.per_trial.pdf", bbox_inches='tight')
        plt.close()
    ## Feedback Types per Trial
    for grp, avgs in group_prs.items():
        x = [n+1 for n in range(len(avgs['min']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_effort')
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Pecent Scalar')
    plt.title('Percent Scalar Feedback')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("percent_scalars.per_trial.pdf", bbox_inches='tight')
        plt.close()

    ## Feedback Types per Trial
    for grp, avgs in group_non.items():
        x = [n+1 for n in range(len(avgs['min']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_effort')
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Num None')
    plt.title('Quantity None')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("no_feedback.per_trial.pdf", bbox_inches='tight')
        plt.close()
    ## Feedback Types per Trial
    for grp, avgs in group_prn.items():
        x = [n+1 for n in range(len(avgs['min']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_effort')
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Pecent None')
    plt.title('Percent No Feedback')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("percent_no_feedback.per_trial.pdf", bbox_inches='tight')
        plt.close()

    ## Violations per Total Effort
    for grp, avgs in group_violations.items():
        steps = [0]
        steps.extend(avgs['len'])
        x = np.cumsum(steps)
        for k in ('min', 'max', 'end'):
            y = [avgs['ini'][0]]
            y.extend(avgs[k])
            plt.plot(x, y, linestyle=styles[k], color=colors[grp], label=grp+'_'+k)
        ymin = [avgs['ini'][0]]
        ymin.extend(avgs['min'])
        ymax = [avgs['ini'][0]]
        ymax.extend(avgs['max'])
        plt.fill_between(x, ymin, ymax, color=colors[grp], alpha=0.2)
    plt.xlabel('Total Effort')
    plt.ylabel('Violations')
    plt.title('Violations by Effort')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("violations.per_cumulative_effort.pdf", bbox_inches='tight')
        plt.close()

    ## Efficiency is Change in Violations between Trials / Amount of Effort for Trial
    ## Violations per Total Effort
    for grp, avgs in group_violations.items():
        steps = [0]
        steps.extend(avgs['len'])
        x = np.cumsum(steps)
        trial_effort = np.array(avgs['len'])
        ymin = None
        ymax = None
        for k in ('min', 'max', 'end'):
            y = [avgs['ini'][0]]
            y.extend(avgs[k])
            L = np.array(y[:-1])
            R = np.array(y[1:])
            efficiency = (L-R)/trial_effort
            if k == 'min':
                ymin = efficiency
            if k == 'max':
                ymax = efficiency
            plt.plot(x[1:], efficiency, linestyle=styles[k], color=colors[grp], label=grp+'_'+k)
        plt.fill_between(x[1:], ymin, ymax, color=colors[grp], alpha=0.2)
    plt.xlabel('Total Effort')
    plt.ylabel('Efficiency')
    plt.title('Improvement Efficiency')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("efficiency.per_cumulative_effort.pdf", bbox_inches='tight')
        plt.close()

    ## Efficiency per Trial
    for grp, avgs in group_violations.items():
        steps = [0]
        steps.extend(avgs['len'])
        x = np.arange(len(avgs['len'])+1)
        trial_effort = np.array(avgs['len'])
        ymin = None
        ymax = None
        for k in ('min', 'max', 'end'):
            y = [avgs['ini'][0]]
            y.extend(avgs[k])
            L = np.array(y[:-1])
            R = np.array(y[1:])
            efficiency = (L-R)/trial_effort
            if k == 'min':
                ymin = efficiency
            if k == 'max':
                ymax = efficiency
            plt.plot(x[1:], efficiency, linestyle=styles[k], color=colors[grp], label=grp+'_'+k)
        plt.fill_between(x[1:], ymin, ymax, color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Efficiency')
    plt.title('Improvement Efficiency')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("efficiency.per_trial.pdf", bbox_inches='tight')
        plt.close()

def plot_violations(violations_list, detailed_violations, save_prefix, show=True):
    """
    Plots violations as a function of trial
    Plots violations over the course of a trial
    """
    if not show:
        plt.ioff()

    ## Plot policy violations afte reach trial has been completed
    trial_violations = []
    for trial in violations_list:
        trial_violations.append(trial[-1][1])
    trials = np.arange(len(violations_list))+1

    plt.plot(trials, trial_violations)
    plt.xlabel('Trial')
    plt.ylabel('Policy Violation Average After Trial')
    plt.title('Average Violation as a function of Trials')
    if show:
        plt.show()
    else:
        plt.savefig(save_prefix+".per_trial.pdf", bbox_inches='tight')
        plt.close()


    ## Plot of evolution of policy violations as trials progress
    offset = 0
    for trial in violations_list:
        ## Column 1 is violations, Column 0 is iterations
        data = np.array(trial)[:,1]
        length = len(data)
        x = np.arange(length) + offset
        offset += length
        print(f'{data}')
        plt.plot(x, data)
    plt.xlabel('Steps')
    plt.ylabel('Policy Violation Average After Timestep')
    plt.title('Violations Over Time')
    if show:
        plt.show()
    else:
        plt.savefig(save_prefix+".per_step.pdf", bbox_inches='tight')
        plt.close()

    ## Plots of detailed policy violations as trials progress,
    ## The violation map is flattened to 1D and shown as a time series
    ## First, create the flattened timeseries matrix of violations
    total_steps = len(detailed_violations)
    det_vio = [x[1] for x in detailed_violations]
    flattened_violations = np.array(det_vio).reshape((total_steps, 50)).T
    plt.matshow(flattened_violations)
    plt.yticks(np.arange(50))
    plt.grid(True, which='both', axis='y', color='r')
    plt.xlabel('Steps')
    plt.ylabel('State')
    plt.title('States in which Policy Violations Occured at Timestep')
    if show:
        plt.show()
    else:
        plt.savefig(save_prefix+".state_violations_per_step.pdf", bbox_inches='tight')
        plt.close()

def compute_demonstration_losses(data, cg, demonstration_losses):
    ## Recompute the demonstration losses from scratch for now (otherwise, could
    ## just backfill based on the length, but that seems a little complicated.
    nrows, ncols = cg.env.world.shape
    acts = cg.env.actions
    grid_map = cg.env.world

    def update_losses(pi, demo):
        for k in demo:
            state, acti = k
            loss = -torch.log(pi[state[0]*ncols+state[1]][acti])
            demo[k].append(loss.cpu().detach().numpy())
        return demo
    
    demo_losses = {k: [] for k in demonstration_losses}
    for trial in data:
        r = trial.reward_estimate
        print(f'Reward Estimate: {r}')
        cg.env.world = trial.grid_map
        cg.set_reward_estimate(r)
        pi, Q, _ = cg.forward()
        demo_losses = update_losses(pi, demo_losses)

        for transition in trial.transitions:
            r = transition.reward_estimate
            print(f'Reawrd Estimate: {r}')
            cg.set_reward_estimate(r)
            pi, Q, _ = cg.forward()
            demo_losses = update_losses(pi, demo_losses)

    ## Confirm that all lists have the same length:
    len_check = None
    all_same_length = True
    for k, v in demo_losses.items():
        if len_check is None:
            len_check = len(v)
        else:
            all_same_length = all_same_length & (len_check == len(v))
        print(k, len(v))
        assert(all_same_length)

    return demo_losses

def compute_distances(data, cg):
    ## TODO: Use tabular? Use epic_sample?
    ## Try to implement on own?
    ## Overall steps to compute the EPIC psuedometric:
    ## 0) The reward functions must be R(s,a,s')
    ## 1) Canonicalize the reward functions
    ##    C_Ds_Da(R)(s,a,s') = R(s,a,s')+E[gR(s',A,S') - R(s,A,S') - gR(S,A,S')]
    ## 2) Compute the Pearson distance between the two canonicalized reward functions
    ##    C1 = C(R_1)(S,A,S'), C2 = C(R_2)(S,A,S')
    ##    where C1 and C2 are dependent random variables that depend on S,A,S' which are drawn IID
    ##    from D_S and D_A.
    ## Use EPIC as part of objective function?
    nrows, ncols = cg.env.world.shape
    acts = cg.env.actions
    grid_map = cg.env.world

    trans_dict = cg.env.flattened_sas_transitions()
    trans_tuple = cg.env.all_sas_transitions(trans_dict)
    reward_models = []
    for trial in data:
        r = trial.reward_estimate
        reward_models.append(GridWorldRewardModel(r, cg.env, cg.gamma, trans_dict, trans_tuple))
        for transition in trial.transitions:
            r = transition.reward_estimate
            reward_models.append(GridWorldRewardModel(r, cg.env, cg.gamma, trans_dict, trans_tuple))

    ## Now we have our list of reward models. Canonicalize pairs of consecutive reward models
    n_models = len(reward_models)
    distances = dict()
    distances['epic'] = [] ## EPIC distance
    distances['l2_features'] = [] ## L2 norm of the features
    distances['l2_features_norm'] = [] ## L2 norm of normalized features
    distances['l2_reward'] = [] ## L2 norm of R(s,a,s')
    distances['l2_reward_norm'] = [] ## L2 norm of normalized R(s,a,s')
    distances['l2_canonical'] = [] ## L2 norm of the canonialized reward
    distances['l2_canonical_norm'] = [] ## L2 norm of the normalized canonicalized reward
    A = len(acts)
    S = nrows*ncols
    idx = (lambda s,a,sp: s*A*S + a*S + sp)
    ## Sample indices
    indices = np.array([idx(s,a,sp) for s,a,sp in trans_tuple])
    for idx0, idx1 in zip(range(0,n_models-1),range(1,n_models)):
        cr0 = reward_models[idx0]
        cr1 = reward_models[idx1]
        distances['epic'].append(cr0.epic_distance(cr1, trans_tuple).cpu().detach().numpy())
        distances['l2_features'].append(l2_norm(cr0.feature_rewards, cr1.feature_rewards).cpu().detach().numpy())
        distances['l2_features_norm'].append(
            l2_norm(
                normalize(cr0.feature_rewards),
                normalize(cr1.feature_rewards)
            ).cpu().detach().numpy()
        )
        distances['l2_reward'].append(
            l2_norm(
                torch.flatten(cr0.full_reward_model),
                torch.flatten(cr1.full_reward_model),
                indices=indices
            ).cpu().detach().numpy()
        )
        distances['l2_reward_norm'].append(
            l2_norm(
                normalize(torch.flatten(cr0.full_reward_model), indices=indices),
                normalize(torch.flatten(cr1.full_reward_model), indices=indices)
            ).cpu().detach().numpy()
        )
        distances['l2_canonical'].append(
            l2_norm(
                torch.flatten(cr0.canonicalized_reward),
                torch.flatten(cr1.canonicalized_reward),
                indices=indices
            ).cpu().detach().numpy()
        )
        distances['l2_canonical_norm'].append(
            l2_norm(
                normalize(torch.flatten(cr0.canonicalized_reward), indices=indices),
                normalize(torch.flatten(cr1.canonicalized_reward), indices=indices)
            ).cpu().detach().numpy()
        )
    return distances

def normalize(a, indices=None):
    norm = None
    v = None
    if indices is None:
        v = a
    else:
        v = a[indices]
    norm = torch.norm(v)
    if norm < 1e-24:
        return v
    else:
        return v/norm

def l2_norm(a, b, indices=None):
    """
    Computes the L2 norm ||a-b|| over the indices specified.

    If indices=None, then ||a-b||_2 is computed
    """
    if indices is None:
        return torch.norm(a-b)
    else:
        c = (a-b)[indices]
        return torch.norm(c)

def plot_distances(distances, label, save_prefix, show=True):
    if not show:
        plt.ioff()

    x = np.arange(len(distances))
    plt.plot(x, distances, label=label)

    plt.xlabel('Steps')
    plt.ylabel(label + ' disance between t and t+1')
    plt.title(label + ' distance over time')
    #plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig(save_prefix+"."+label+"_distance.per_step.pdf", bbox_inches='tight')
        plt.close()


def plot_losses(demonstration_losses, save_prefix, show=True):
    """
    Plots demo loss as a function of timesteps
    """
    if not show:
        plt.ioff()

    avg_loss = None
    N = len(demonstration_losses)
    for k, v in demonstration_losses.items():
        if avg_loss is not None:
            avg_loss[:] = avg_loss[:] + np.array(v)
        else:
            avg_loss = np.array(v)
        x = np.arange(len(v))
        plt.plot(x, v, label=str(k))
    avg_loss[:] = avg_loss[:] / N
    x = np.arange(len(avg_loss))
    plt.plot(x, avg_loss, label='avg', linewidth=1.5)

    #for k, v in demonstration_losses.items():
    #    x = np.arange(len(v))
    #    plt.plot(x[-1], v[-1], str(k))
    #x = np.arange(len(avg_loss))
    #plt.plot(x[-1], avg_loss[-1], 'avg')

    plt.xlabel('Steps')
    plt.ylabel('Loss After Timestep')
    plt.title('Losses of Demostrated Examples')
    #plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig(save_prefix+".loss.per_step.pdf", bbox_inches='tight')
        plt.close()
    

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', help='"train", "test", or "plot" mode', required=True, type=str,
        choices=["train","test","plot","extract_policies", "extract_reward", "single_group_metric",
                "plot_reward"])
    parser.add_argument('--evalmetric', help='"dgs", "dpv", "sgs", "spv", "rew"', type=str,
        choices=["dgs","dpv","sgs","spv", "rew"], default=None)
    parser.add_argument('--saveplot', type=str)
    parser.add_argument('--plottitle', type=str)
    parser.add_argument('--xlabel', type=str)
    parser.add_argument('--ylabel', type=str)
    parser.add_argument('--ylo', type=float)
    parser.add_argument('--yhi', type=float)
    parser.add_argument('--scale', type=float)
    parser.add_argument('--dataset', help='path to dataset', type=str)
    parser.add_argument('--style', help='plot style', type=str)
    parser.add_argument('--prepopulate', help='prepopulate with action feedback',
        dest='prepopulate', action='store_true')
    parser.add_argument('--feedback_policy',
        default="human", choices=["human","action", "evaluative", "mixed",
            "r1_evaluative", "ranked_evaluative", "raw_evaluative", "policy_evaluation", "ranked_policy_evaluation",
            "action_path_cost"],
        help='specify the feedback policy to use')
    parser.add_argument('--hide', help='hide plots, used for autosaving plots',
        dest='hide', action='store_true')
    parser.add_argument('--noforce', help='Agent decides whether to request feedback',
        dest='noforce', action='store_true')
    parser.add_argument('--inputs', help='paths to dataset files used for plotting', nargs='+')
    parser.add_argument('--groupings', help='keyword groupings', nargs='+')
    parser.add_argument('--lastonly', help='plot flag stating to compute metric only for the last timestep of each episode', dest='lastonly', action='store_true')
    parser.add_argument('--seed', help='specifies a random seed', type=int,
        default=np.random.default_rng().integers(1000000))
    parser.add_argument('--episodes', type=int,
        default=10, help='determines number of episodes for which to train the agent for. Default is 10')
    parser.add_argument('--gamma', type=float,
        default=0.90, help='discount factor to use')
    parser.add_argument('--alpha', type=float,
        default=0.001, help='learning rate to use')
    parser.add_argument('--group_alphas', nargs='+', help='For hyperparameter sweeps', default=None)
    parser.add_argument('--group_gammas', nargs='+', help='For hyperparameter sweeps', default=None)
    parser.add_argument('--mixed_strat', nargs='+', help='Pair of strategies to consider', default=["action_path_cost", "r1_evaluative"])
    parser.add_argument('--mixed_percent', type=float, help='Probability of following first strat in mixed strat on each timestep', default=0.5)
    parser.add_argument('--violation_set', nargs='+', help='List the violating states', default=[1,2,3], choices=range(9), type=int)
    world_choices =[x for x in range(Worlds.max_idx+1)]
    parser.add_argument('--world', type=int,
        help='Integer index of the world to train in', default=None,
        choices=world_choices)
    parser.add_argument('--categories', type=int,
        help="Integer number of categories in the world", default=None)
    return parser.parse_args()

def prepopulate(cg):
    r, c = next(cg.env.random_start_state())
    g = cg.env.get_goal_state()
    prepop_trial_data = Trial((r,c), g, cg.current_reward_estimate(), cg.env.get_world())
    prepop_trial_data.register_feedback_type(cg.env.SCALAR_FEEDBACK)
    prepop_trial_data.register_feedback_type(cg.env.ACTION_FEEDBACK)

    nrows, ncols = cg.env.world.shape
    acts = cg.env.actions
    grid_map = cg.env.world

    steps = 0
    for r in range(nrows):
        for c in range(ncols):
            action, local_policy = cg.chooseAction(r, c)
            feedback_str = cg.env.acquire_feedback(action, r, c, source_is_human=False)
            feedback = None
            ## Classify the feedback
            feedback_type = cg.env.classify_feedback(feedback_str)
            if feedback_type == cg.env.ACTION_FEEDBACK:
                print("Action Feedback Provided")
                #Collect Trajectories
                trajacts, trajcoords = cg.env.feedback_to_demonstration(feedback_str, r, c)
                feedback = trajacts[0]
            prepop_trial_data.add_transition(
                steps, cg.current_reward_estimate(),
                (r,c), action, local_policy, feedback_type, feedback
            )
            steps += 1

    return prepop_trial_data

if __name__ == '__main__':
    """
    Plotting has been decomposed into 3 steps:
        (1) Policy extraction
        (2) Single group metric computations
        (3) Plotting, which plots the specified metrics together
    """
    args = parse_args()
    all_training_data = None
    demonstration_losses = None
    ## Select a world
    if args.world is None:
        print(f"Need to specify a world!")
        exit()
    if args.categories is None:
        print(f"Need to number of categories in the world!")
        exit()
    idx = args.world
    grid_maps, state_starts, viz_starts = Worlds.define_worlds()
    Worlds.categories = [i for i in range(args.categories)]
    env = Environment(grid_maps[idx], state_starts[idx], viz_starts[idx], Worlds.categories)
    gamma = args.gamma
    alpha = args.alpha
    cg = ComputationGraph(env)
    cg.set_gamma(gamma)
    cg.set_alpha(alpha)
    prepop_trial_data = None
    show_plots = True
    force_feedback = True
    if args.hide:
        show_plots = False
        if args.dataset is None:
            print(f"Save prefix must be specified if hiding plots.")
            exit()

    if args.noforce:
        force_feedback = False

    if args.feedback_policy == "mixed":
        if args.mixed_strat is None:
            print(f"Mixed strategy components must be specified if a mixed strategy is used")
            exit()
        ## Need to put a check here to confirm that strategy types present here are valid
    elif "extract_" in args.mode:
        ## Policy extract is the only portion of plotting that requires hyperparameters to be set
        plt.rcParams.update({'font.size': 20})
        if args.inputs is None or len(args.inputs) == 0:
            print(f'input datasets must be specified')
            exit()

        d_hypers = dict()
        grp_idx = 0
        for grp in args.groupings:
            grp_hyper = dict()
            if args.group_alphas is not None and len(args.group_alphas) == len(args.groupings):
                grp_hyper['alpha'] = float(args.group_alphas[grp_idx])
            if args.group_gammas is not None and len(args.group_gammas) == len(args.groupings):
                grp_hyper['gamma'] = float(args.group_gammas[grp_idx])
            grp_idx = grp_idx + 1
            d_hypers[grp] = grp_hyper

        for dataset in args.inputs:
            if not os.path.isfile(dataset):
                print(f"Dataset path is not a file.")
                exit()
            with open(dataset, 'rb') as f:
                print(f"Loading... {dataset}")
                all_training_data = pickle.load(f)
                ## Check if we have the loss saved in the pickle file
                try:
                    demonstration_losses = pickle.load(f)
                except EOFError:
                    print("No demonstration losses in this dataset")
            ## Fix for missing data
            for t in all_training_data:
                if hasattr(t, "feedback_indices"):
                    if cg.env.NO_FEEDBACK not in t.feedback_indices:
                        t.feedback_indices[cg.env.NO_FEEDBACK] = []
                else:
                    setattr(t, "feedback_indices", dict())
                    t.feedback_indices[cg.env.NO_FEEDBACK] = []
                    t.feedback_indices[cg.env.ACTION_FEEDBACK] = []
                    t.feedback_indices[cg.env.SCALAR_FEEDBACK] = []

            ## Set the hyperparameters for this dataset if they were provided
            for grp in args.groupings:
                if grp in dataset:
                    cg.set_hyperparameters(d_hypers[grp])

            ## Policy extract is the only portion of plotting that requires hyperparameters to be set
            if args.mode == "extract_policies":
                policies = extract_policies(all_training_data, cg, last_only=args.lastonly)
                with open(f"{dataset}.policy","wb") as f:
                    pickle.dump(policies, f)
            elif args.mode == "extract_reward":
                rewards = extract_rewards(all_training_data, cg, last_only=args.lastonly)
                with open(f"{dataset}.rewards","wb") as f:
                    pickle.dump(rewards, f)
            else:
                raise ValueError("Extraction must be of either rewards or policies")
        exit()
    elif args.mode == "single_group_metric":
        """
        Computes the metrics for a SINGLE group of data, and saves it

        A group consists of N seeds (trials), where all experimental parameters
        are held constant. We load in the policies for each trial of M episodes.
        The policy for the n'th seed and m'th episode is given by p_nm

        An evaluation metric f() is computed according to the args.evalmetric key for
        each policy

        The data will be tabulated in the following format:

        An NxM numpy array is created, with f(p_nm) in each entry
        [ f(p_nm) ]

        """
        if args.evalmetric is None:
            print("Evaluation Metric must be specified:")
            print("  dgs = deterministic goal success")
            print("  dpv = deterministic policy violations")
            print("  sgs = stochastic goal success")
            print("  spv = stochastic policy violations")
            print("  rew = rewards")
            exit()
        plt.rcParams.update({'font.size': 20})
        if args.inputs is None or len(args.inputs) == 0:
            print(f'input datasets must be specified')
            exit()

        if args.dataset is None:
            print("Need to specify base_name to save dataset as")

        result_arr = None
        seed_idx = 0

        ## Support step-based plotting in addition to episode-based
        ## plotting.
        max_length = 0
        for dataset in args.inputs:
            if not os.path.isfile(dataset):
                print(f"Dataset path is not a file.")
                exit()
            with open(dataset, 'rb') as f:
                print(f"Loading... {dataset}")
                policies = pickle.load(f)
            if len(policies) > max_length:
                max_length = len(policies)

        ## For step-based each seed might be a different length
        ## If the length is smaller than the max_length, then we
        ## backfill with the value of the last element
        if result_arr is None:
            seeds = len(args.inputs)
            episodes = len(policies)
            if args.evalmetric == "rew":
                reward_dim = len(policies[0])
                result_arr = np.zeros((seeds, max_length, reward_dim))
            else:
                result_arr = np.zeros((seeds, max_length))

        for dataset in args.inputs:
            if not os.path.isfile(dataset):
                print(f"Dataset path is not a file.")
                exit()
            with open(dataset, 'rb') as f:
                print(f"Loading... {dataset}")
                policies = pickle.load(f)
            
            episodes = len(policies)
            result = None

            if   args.evalmetric == "dgs":
                result = save_policy_metrics(policies, cg, which="dgs")
            elif args.evalmetric == "dpv":
                result = save_policy_metrics(policies, cg, which="dpv")
            elif args.evalmetric == "sgs":
                result = stoch_goal_success(policies, cg)
            elif args.evalmetric == "spv":
                result = stoch_policy_violations(policies, cg, args.violation_set)
            elif args.evalmetric == "rew":
                result = np.array(policies)
            else:
                exit()
            if episodes == max_length:
                result_arr[seed_idx] = result
            else:
                result_arr[seed_idx, 0:episodes] = result
                result_arr[seed_idx, episodes:] = result[-1]
                
            seed_idx += 1

        with open(f"{args.dataset}.{args.evalmetric}","wb") as f:
            pickle.dump(result_arr, f)
    elif args.mode == "plot_reward":
        """
        Special plotting for reward
        """
        plt.rcParams.update({'font.size': 20})
        if args.inputs is None or len(args.inputs) == 0:
            print(f'input "single_group_metric"s must be specified')
            exit()
        ## Set plotting styles
        styles = {
            'avg': '-',
            'med': '-.'
        }
        #color_list = ['blue','green', 'red', 'cyan', 'magenta', 'yellow', 'black']
        #color_list = ['magenta','gold', 'green', 'blue', 'red', 'cyan', 'black']
        #default order: c0=blue, c1=orange, c2=green, c3=red, c4=violet, c5=brown, c6=pink
        #perceptual: green blue violet pink orange red brown
        color_list = ['#6bd0f3', '#077ccc', '#075791', '#AA0000', 'black', 'C3', 'C5', 'C7', 'C8', 'C9', 'black']
        colors = {k:v for k,v in zip(args.inputs, color_list)}

        for dataset in args.inputs:
            if not os.path.isfile(dataset):
                print(f"Dataset path is not a file.")
                exit()
            with open(dataset, 'rb') as f:
                print(f"Loading... {dataset}")
                data = pickle.load(f)

            ## Compute all the statistics of this dataaset
            data_avg = np.mean(data,axis=0)
            data_std = np.std(data, axis=0)
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
            data_med = np.median(data, axis=0)
            data_upper = np.quantile(data, axis=0, q=0.84, interpolation='linear')
            data_lower = np.quantile(data, axis=0, q=0.16, interpolation='linear')
            x = [n+1 for n in range(len(data_avg))]

            for i in range(len(data[0,0])):
                if args.style == "std":
                    plt.fill_between(x,
                        data_avg.T[i] - data_std.T[i]*args.scale,
                        data_avg.T[i] + data_std.T[i]*args.scale,
                        color=color_list[i],
                        alpha=0.1)
                elif args.style == "quantile":
                    plt.fill_between(x,
                        data_lower.T[i],
                        data_upper.T[i],
                        color=color_list[i],
                        alpha=0.1)
                plt.plot(x,
                    data_avg.T[i],
                    linestyle=styles["avg"],
                    color=color_list[i],
                    label=f"{i}")

        if args.ylo is not None and args.yhi is not None:
            plt.ylim([args.ylo, args.yhi])
        plt.xlabel(args.xlabel)
        plt.ylabel(args.ylabel)
        plt.savefig(args.saveplot, bbox_inches='tight')
        plt.close()
    elif args.mode == "plot":
        """
        Plot takes in a set of "single_group_metrics" and plots them
        all on the same plot in the order they are presented

        Each "single_group_metric" is assumed to the following:
            - A N x M numpy array, where N = seeds, M = episodes
        """
        plt.rcParams.update({'font.size': 20})
        if args.inputs is None or len(args.inputs) == 0:
            print(f'input "single_group_metric"s must be specified')
            exit()
        ## Set plotting styles
        styles = {
            'avg': '-',
            'med': '-.'
        }
        #color_list = ['blue','green', 'red', 'cyan', 'magenta', 'yellow', 'black']
        #color_list = ['magenta','gold', 'green', 'blue', 'red', 'cyan', 'black']
        #default order: c0=blue, c1=orange, c2=green, c3=red, c4=violet, c5=brown, c6=pink
        #perceptual: green blue violet pink orange red brown
        color_list = ['C2', 'C0', 'C4', 'C6', 'C1', 'C3', 'C5', 'C7', 'C8', 'C9', 'black']
        colors = {k:v for k,v in zip(args.inputs, color_list)}

        for dataset in args.inputs:
            if not os.path.isfile(dataset):
                print(f"Dataset path is not a file.")
                exit()
            with open(dataset, 'rb') as f:
                print(f"Loading... {dataset}")
                data = pickle.load(f)

            ## Compute all the statistics of this dataaset
            data_avg = np.mean(data,axis=0)
            data_std = np.std(data, axis=0)
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
            data_med = np.median(data, axis=0)
            data_upper = np.quantile(data, axis=0, q=0.84, interpolation='linear')
            data_lower = np.quantile(data, axis=0, q=0.16, interpolation='linear')
            x = [n+1 for n in range(len(data_avg))]

            if args.style == "std":
                plt.fill_between(x, data_avg - data_std*args.scale, data_avg + data_std*args.scale, color=colors[dataset], alpha=0.1)
            #plt.fill_between(x, data_min, data_max, color=colors[dataset], alpha=0.05)
            elif args.style == "quantile":
                plt.fill_between(x, data_lower, data_upper, color=colors[dataset], alpha=0.1)
            plt.plot(x, data_avg, linestyle=styles["avg"], color=colors[dataset])

        if args.ylo is not None and args.yhi is not None:
            plt.ylim([args.ylo, args.yhi])
        plt.xlabel(args.xlabel)
        plt.ylabel(args.ylabel)
        plt.savefig(args.saveplot, bbox_inches='tight')
        plt.close()
    else:
        print(f"Mode must be train or test")
        exit()

    exit()
    plt.rcParams.update({'font.size': 20})
    all_violations, detailed_violations = compute_violations(all_training_data, cg)
    plot_violations(all_violations, detailed_violations, args.dataset, show=show_plots)
    if demonstration_losses is not None:
        demo_losses = compute_demonstration_losses(all_training_data, cg, demonstration_losses)
        plot_losses(demo_losses, args.dataset, show=show_plots)

    distances = compute_distances(all_training_data, cg)
    for k, v in distances.items():
        plot_distances(v, k, args.dataset, show=show_plots)
