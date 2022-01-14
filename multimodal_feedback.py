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


def train(episodes, cg, prepop_trial_data=None, force_feedback=True,
    source_is_human=True, 
    feedback_policy_type="human", 
    mixed_strat=None, 
    mixed_percent=0.5,
    violation_set={1,2,3}):
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
                feedback_str = cg.env.acquire_feedback(action, r, c,
                    source_is_human=source_is_human,
                    feedback_policy_type=feedback_policy_type,
                    agent_cg=cg,
                    mixed_strat=mixed_strat,
                    mixed_percent=mixed_percent,
                    violation_set=violation_set)
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
    parser.add_argument('--mode', help='"train", "test", or "plot" mode', required=True, type=str)
    parser.add_argument('--dataset', help='path to dataset', type=str)
    parser.add_argument('--prepopulate', help='prepopulate with action feedback',
        dest='prepopulate', action='store_true')
    parser.add_argument('--feedback_policy',
        default="human", choices=["human","action", "evaluative", "mixed",
            "r1_evaluative", "ranked_evaluative", "raw_evaluative", "policy_evaluation", "ranked_policy_evaluation",
            "action_path_cost", "affine_policy_evaluation", "heuristic_action_eval"],
        help='specify the feedback policy to use')
    parser.add_argument('--hide', help='hide plots, used for autosaving plots',
        dest='hide', action='store_true')
    parser.add_argument('--noforce', help='Agent decides whether to request feedback',
        dest='noforce', action='store_true')
    parser.add_argument('--inputs', help='paths to dataset files used for plotting', nargs='+')
    parser.add_argument('--groupings', help='keyword groupings', nargs='+')
    parser.add_argument('--lastonly', help='plot flag stating to compute metric only for the last timestep of each episode',
        dest='lastonly', action='store_true')
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
    parser.add_argument('--bellman_update', choices=["softmax", "max", "min", "mean"], default="softmax")
    
    world_choices =[x for x in range(Worlds.max_idx+1)]
    parser.add_argument('--mixed_percent', type=float, help='Probability of following first strat in mixed strat on each timestep', default=0.5)
    parser.add_argument('--world', type=int,
        help='Integer index of the world to train in', default=0,
        choices=world_choices)
    parser.add_argument('--categories', type=int,
        help="Integer number of categories in the world", default=None)
    parser.add_argument('--violation_set', nargs='+', help='List the violating states', default=None, choices=range(9), type=int)
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
    if args.violation_set is None:
        print(f"Need to specify violating states!")
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
    cg.set_fmax(args.bellman_update)
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

    if args.mode == "train":
        if args.prepopulate:
            print("Prepopulating trial data")
            prepop_trial_data = prepopulate(cg)
        source_is_human = (args.feedback_policy == "human")
        episodes = args.episodes
        all_training_data, demonstration_losses = train(episodes, cg, prepop_trial_data,
            force_feedback, source_is_human,
            args.feedback_policy, args.mixed_strat,
            args.mixed_percent, args.violation_set)
    elif args.mode == "test":
        if args.dataset is None:
            print(f"Dataset path must be specified.")
            exit()

        if not os.path.isfile(args.dataset):
            print(f"Dataset path is not a file.")
            exit()

        with open(args.dataset, 'rb') as f:
            all_training_data = pickle.load(f)
            ## Check if we have the loss saved in the pickle file
            try:
                demonstration_losses = pickle.load(f)
            except EOFError:
                print("No demonstration losses in this dataset")
    elif args.mode == "plot":
        plt.rcParams.update({'font.size': 20})
        if args.inputs is None or len(args.inputs) == 0:
            print(f'input datasets must be specified')
            exit()
        ## Plot violation averages, maxes, and mins
        group_avg = {grp:None for grp in args.groupings} ## Violations
        group_gsr = {grp:None for grp in args.groupings} ## Goal Success Rate
        group_eff = {grp:None for grp in args.groupings} ## Effort
        group_act = {grp:None for grp in args.groupings} ## Action Effort
        group_sca = {grp:None for grp in args.groupings} ## Scalar Effort
        group_non = {grp:None for grp in args.groupings} ## None   Effort
        group_pra = {grp:None for grp in args.groupings} ## Percent Action
        group_prs = {grp:None for grp in args.groupings} ## Percent Scalar
        group_prn = {grp:None for grp in args.groupings} ## Percent None
        group_num = {grp:0 for grp in args.groupings}

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

            all_violations, detailed_violations, goal_success_rates = compute_policy_eval_metrics(all_training_data, cg, last_only=args.lastonly)
            # all_violations, detailed_violations = compute_violations(all_training_data, cg)
            violations_dict = min_mean_max_violations(all_violations)
            # goal_success_rates = compute_goal_success(all_training_data, cg)
            goal_success_dict = min_mean_max_goal_success(goal_success_rates)
            trial_efforts, effort_dist = compute_effort(all_training_data, cg)
            ## Percentage Action / Scalar
            num_action = np.array(effort_dist[cg.env.ACTION_FEEDBACK])
            num_scalar = np.array(effort_dist[cg.env.SCALAR_FEEDBACK])
            num_none   = np.array(effort_dist[cg.env.NO_FEEDBACK])
            total_num  = num_action + num_scalar + num_none
            per_action = num_action / total_num
            per_scalar = num_scalar / total_num
            per_none   = num_none   / total_num

            ## Determine which group this dataset belongs to
            for grp, grp_avg in group_avg.items():
                if grp in dataset:
                    if grp_avg is None:
                        group_avg[grp] = dict(violations_dict)
                        group_num[grp] = 1
                        group_gsr[grp] = dict(goal_success_dict)
                        group_eff[grp] = {
                            'min': np.array(trial_efforts),
                            'avg': np.array(trial_efforts),
                            'max': np.array(trial_efforts)
                        }
                        group_act[grp] = {
                            'min': np.array(num_action),
                            'avg': np.array(num_action),
                            'max': np.array(num_action)
                        }
                        group_sca[grp] = { 
                            'min': np.array(num_scalar),
                            'avg': np.array(num_scalar),
                            'max': np.array(num_scalar)
                        }
                        group_non[grp] = { 
                            'min': np.array(num_none),
                            'avg': np.array(num_none),
                            'max': np.array(num_none)
                        }
                        group_pra[grp] = {
                            'min': np.array(per_action),
                            'avg': np.array(per_action),
                            'max': np.array(per_action)
                        }
                        group_prs[grp] = { 
                            'min': np.array(per_scalar),
                            'avg': np.array(per_scalar),
                            'max': np.array(per_scalar)
                        }
                        group_prn[grp] = { 
                            'min': np.array(per_none),
                            'avg': np.array(per_none),
                            'max': np.array(per_none)
                        }
                    else:
                        ## Update running averages
                        n = group_num[grp] + 1
                        group_num[grp] = n
                        if not args.lastonly:
                            ## grp_avg contains the previous average
                            for k in ('min', 'max', 'end', 'ini', 'len'):
                                for idx, v in enumerate(grp_avg[k]):
                                    group_avg[grp][k][idx] = (v*(n-1) + violations_dict[k][idx])/n
                                for idx, v in enumerate(group_gsr[grp][k]):
                                    group_gsr[grp][k][idx] = (v*(n-1) + goal_success_dict[k][idx])/n
                        else:
                            k = 'end'
                            ## Make sure end is average of seeds
                            for idx, v in enumerate(grp_avg[k]):
                                group_avg[grp][k][idx] = (v*(n-1) + violations_dict[k][idx])/n
                            for idx, v in enumerate(group_gsr[grp][k]):
                                group_gsr[grp][k][idx] = (v*(n-1) + goal_success_dict[k][idx])/n
                            k = 'len'
                            ## Make sure end is average of seeds
                            for idx, v in enumerate(grp_avg[k]):
                                group_avg[grp][k][idx] = (v*(n-1) + violations_dict[k][idx])/n
                            for idx, v in enumerate(group_gsr[grp][k]):
                                group_gsr[grp][k][idx] = (v*(n-1) + goal_success_dict[k][idx])/n
                            k = 'ini'
                            ## Make sure end is average of seeds
                            for idx, v in enumerate(grp_avg[k]):
                                group_avg[grp][k][idx] = (v*(n-1) + violations_dict[k][idx])/n
                            for idx, v in enumerate(group_gsr[grp][k]):
                                group_gsr[grp][k][idx] = (v*(n-1) + goal_success_dict[k][idx])/n
                            k = 'min'
                            ## Make sure min is min of seeds
                            for idx, v in enumerate(grp_avg[k]):
                                group_avg[grp][k][idx] = min(v, violations_dict[k][idx])
                            for idx, v in enumerate(group_gsr[grp][k]):
                                group_gsr[grp][k][idx] = min(v, goal_success_dict[k][idx])
                            k = 'max'
                            ## Make sure min is min of seeds
                            for idx, v in enumerate(grp_avg[k]):
                                group_avg[grp][k][idx] = max(v, violations_dict[k][idx])
                            for idx, v in enumerate(group_gsr[grp][k]):
                                group_gsr[grp][k][idx] = max(v, goal_success_dict[k][idx])

                        ## Update group effort statistics: Update average effort per trial
                        ## Trial effort includes only Action and Scalar feedback counts
                        for idx, v in enumerate(group_eff[grp]['avg']):
                            group_eff[grp]['avg'][idx] = (v*(n-1) + trial_efforts[idx])/n
                        for idx, v in enumerate(group_eff[grp]['min']):
                            group_eff[grp]['min'][idx] = min(v, trial_efforts[idx])
                        for idx, v in enumerate(group_eff[grp]['max']):
                            group_eff[grp]['max'][idx] = max(v, trial_efforts[idx])

                        ## Update Effort distribution: number of action feedback per trial
                        for idx, v in enumerate(group_act[grp]['avg']):
                            group_act[grp]['avg'][idx] = (v*(n-1) + effort_dist[cg.env.ACTION_FEEDBACK][idx])/n
                        for idx, v in enumerate(group_act[grp]['min']):
                            group_act[grp]['min'][idx] = min(v, effort_dist[cg.env.ACTION_FEEDBACK][idx])
                        for idx, v in enumerate(group_act[grp]['max']):
                            group_act[grp]['max'][idx] = max(v, effort_dist[cg.env.ACTION_FEEDBACK][idx])

                        ## Update Effort distribution: number of action feedback per trial
                        for idx, v in enumerate(group_pra[grp]['avg']):
                            group_pra[grp]['avg'][idx] = (v*(n-1) + per_action[idx])/n
                        for idx, v in enumerate(group_pra[grp]['min']):
                            group_pra[grp]['min'][idx] = min(v, per_action[idx])
                        for idx, v in enumerate(group_pra[grp]['max']):
                            group_pra[grp]['max'][idx] = max(v, per_action[idx])

                        ## Update Effort Distribution: number of scalar feedback per trial
                        for idx, v in enumerate(group_sca[grp]['avg']):
                            group_sca[grp]['avg'][idx] = (v*(n-1) + effort_dist[cg.env.SCALAR_FEEDBACK][idx])/n
                        for idx, v in enumerate(group_sca[grp]['min']):
                            group_sca[grp]['min'][idx] = min(v, effort_dist[cg.env.SCALAR_FEEDBACK][idx])
                        for idx, v in enumerate(group_sca[grp]['max']):
                            group_sca[grp]['max'][idx] = max(v, effort_dist[cg.env.SCALAR_FEEDBACK][idx])

                        ## Update Effort Distribution: number of scalar feedback per trial
                        for idx, v in enumerate(group_prs[grp]['avg']):
                            group_prs[grp]['avg'][idx] = (v*(n-1) + per_scalar[idx])/n
                        for idx, v in enumerate(group_prs[grp]['min']):
                            group_prs[grp]['min'][idx] = min(v, per_scalar[idx])
                        for idx, v in enumerate(group_prs[grp]['max']):
                            group_prs[grp]['max'][idx] = max(v, per_scalar[idx])

                        ## Update Effort Distribution: number of no feedback per trial
                        for idx, v in enumerate(group_non[grp]['avg']):
                            group_non[grp]['avg'][idx] = (v*(n-1) + effort_dist[cg.env.NO_FEEDBACK][idx])/n
                        for idx, v in enumerate(group_non[grp]['min']):
                            group_non[grp]['min'][idx] = min(v, effort_dist[cg.env.NO_FEEDBACK][idx])
                        for idx, v in enumerate(group_non[grp]['max']):
                            group_non[grp]['max'][idx] = max(v, effort_dist[cg.env.NO_FEEDBACK][idx])

                        ## Update Effort Distribution: number of no feedback per trial
                        for idx, v in enumerate(group_prn[grp]['avg']):
                            group_prn[grp]['avg'][idx] = (v*(n-1) + per_none[idx])/n
                        for idx, v in enumerate(group_prn[grp]['min']):
                            group_prn[grp]['min'][idx] = min(v, per_none[idx])
                        for idx, v in enumerate(group_prn[grp]['max']):
                            group_prn[grp]['max'][idx] = max(v, per_none[idx])
        ## Adjust the baseline (remove the first episode)
        if 'baseline' in group_avg:
            for k in ('min', 'max', 'end', 'ini', 'len'):
                group_avg['baseline'][k] = group_avg['baseline'][k][1:]
                group_gsr['baseline'][k] = group_gsr['baseline'][k][1:]
            for k in ('min', 'avg', 'max'):
                group_eff['baseline'][k] = group_eff['baseline'][k][1:]
                group_act['baseline'][k] = group_act['baseline'][k][1:]
                group_sca['baseline'][k] = group_sca['baseline'][k][1:]
                group_non['baseline'][k] = group_sca['baseline'][k][1:]
                group_pra['baseline'][k] = group_pra['baseline'][k][1:]
                group_prs['baseline'][k] = group_prs['baseline'][k][1:]
                group_prn['baseline'][k] = group_prs['baseline'][k][1:]
        plot_group_violations(
            group_avg, group_gsr, group_eff, 
            group_act, group_sca, group_non,
            group_pra, group_prs, group_prn,
            show=show_plots
        )

        exit()
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
