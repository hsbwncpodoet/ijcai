from functools import reduce
import copy
from .world import Worlds
import torch
import numpy as np
from computation_graph import ComputationGraph
from grid_world_search import GridWorldProblem, GridWorldSearch

class Environment:

    def __init__(self, world, start, viz_start, categories):
        self.world = world
        self.state_start = start
        self.viz_start = viz_start
        self.categories = categories
        self.ncategories = len(categories)
        self.actions = [(-1,0), (0,1), (1,0), (0,-1), (0,0)]
        self.act_map = Worlds.act_map
        self.act_name = Worlds.act_name
        self.dtype = torch.float32
        self.action_feedback_map = {
            "w": 0,
            "d": 1,
            "s": 2,
            "a": 3,
            "stay": 4
        }

        self.SCALAR_FEEDBACK = 100
        self.ACTION_FEEDBACK = 200
        self.NO_FEEDBACK = 300

        self.need_simulated_evaluative_feedback = True
        self.sim_cg = None
        self.sim_advantage = None
        self.transition_matrix = None
        self.matrix_map = None
        self.rng = np.random.default_rng()
        self._costs = None
        self._paths = None
        self._problem = None
        self.idx2act = ["w","d","s","a","stay"]

        self.violation_matrix = None

    def flattened_sas_transitions(self):
        """
        Returns 1D version of (s,a,s') map
        """
        nrows, ncols = self.world.shape
        state_idx = (lambda r, c: r*ncols + c)
        in_bounds = (lambda r, c: (0 <= r and r < nrows) and (0 <= c and c < ncols))
        transitions = dict()
        for r in range(nrows):
            for c in range(ncols):
                for a in range(len(self.actions)):
                    action = self.actions[a]
                    s = state_idx(r,c)
                    rp = r + action[0]
                    cp = c + action[1]
                    if in_bounds(rp, cp):
                        sp = state_idx(rp,cp)
                        if s not in transitions:
                            transitions[s] = {a: sp}
                        else:
                            transitions[s][a] = sp
        return transitions

    def all_sas_transitions(self, transitions):
        print(transitions)
        """
        Given a transitions dict, returns the possible tuples (s,a,s')
        """
        return tuple( tuple((s,a,sp))
            for s, asp in transitions.items()
            for a, sp in asp.items()
        )

    def get_world(self):
        return copy.deepcopy(self.world)

    def inform_human(self, action_idx):
        """
        The environment informs the human what action the agent is planning to take
        """
        print(f"The agent plans to do action: {self.act_name[action_idx]}")

    def request_feedback_from_human(self):
        """
        The agent will request a specific feedback type from the human
        """
        pass

    def acquire_feedback(self, action_idx, r, c,
            source_is_human=True,
            feedback_policy_type="action",
            agent_cg=None,
            mixed_strat=None,
            mixed_percent=0.50,
            violation_set={1,2,3}):
        """
        Acquires feedback from a source

        action_idx = index into the env.actions array such that env.actions[action_idx] is the index
            the agent plans on making
        r, c = row and colum position of the agent
        source_is_human = boolean indicating whether a human will give the feedback, or if an automated
            teacher will provide the feedback
        feedback_policy_type = if an automated teaching/feedback strategy is used, what is that strategy?
        agent_cg = for some teaching / feedback strategies, it's assumed the teacher has formed some estimate
            or model of the agent's policy, based on observing the agent's behavior over time. We pass in the
            the agent's computational graph in order to automated teaching strategies to simplify this modeling
            process.  If a teaching strategy should not have access to the agent's policy, then pass in None
        """
        def is_number(f):
            """
            This function is here to support raw numerical feedback (from teaching/feedback automation).
            """
            try:
                ffff = float(f)
            except ValueError:
                return False
            except TypeError:
                return False
            return True

        valid_feedback = {"-2","-1","0","1","2","w","a","s","d","stay", "x"}
        feedback_str = None
        valid = True
        while feedback_str not in valid_feedback and not is_number(feedback_str):
            if source_is_human:
                if not valid: print("Invalid feedback specified")
                print("Feedback Options: Action Options:  w(UP), d(RIGHT), s(DOWN), a(LEFT), stay(STAY)")
                print("Feedback Options: Scalar Options:  -2,  -1,  0,  1,  2")
                feedback_str = input("Human Feedback: ")
            else:
                feedback_str = self.feedback_source(action_idx, r, c, feedback_policy_type, agent_cg, mixed_strat, mixed_percent, violation_set)
            valid = False
        return feedback_str

    def classify_feedback(self, feedback):
        """
        Classifies the feedback into a feedback category
        """
        print(f"Recieved: {feedback}")
        if feedback in "-2 -1 0":
            return self.SCALAR_FEEDBACK
        else:
            try:
                f = float(feedback)
                return self.SCALAR_FEEDBACK
            except ValueError:
                ## Couldn't convert feedback to a float. Assuming ACTION_FEEDBACK
                return self.ACTION_FEEDBACK

    def feedback_to_scalar(self, feedback):
        """
        Converts feedback to scalar
        """
        return float(feedback)

    def feedback_to_demonstration(self, feedback, r, c):
        """
        Converts feedback to trajact and trajcoord
        """
        return [self.action_feedback_map[feedback]], [(r,c)]

    def feedback_source(self, action, r, c, feedback_policy_type, agent_cg, mixed_strat, mixed_percent, violation_set):
        """
        A simulator for providing feedback.
        This function is called from acquire_feedback()

        action - idx in to the self.actions so that self.actions[action] corresponds with the
            action the agent wishes to take
        r, c - gridworld coordinates row, col
        feedback_policy_type - string, this comes in as a program argument
        agent_cg = for some teaching / feedback strategies, it's assumed the teacher has formed some estimate
            or model of the agent's policy, based on observing the agent's behavior over time. We pass in the
            the agent's computational graph in order to automated teaching strategies to simplify this modeling
            process.  If a teaching strategy should not have access to the agent's policy, then pass in None
        """
        nrows, ncols = self.world.shape
        ## TODO: Implement a feedback simulator to simulate giving feedback
        ## NOTE: For now, we've implemented optimal feedback for each grid location
        if feedback_policy_type == "action":
            """
            Ideally generalize this to optimal actions for arbitrary worlds other than world 1
            Currently, this is relegated to single time-step trajectories
            """
            ## Action only
            feedback_map = (
                ("s","s","a","d","s","s","s","s","d","stay"),
                ("s","s","s","d","d","d","d","d","d","w"),
                ("d","d","d","d","d","d","d","d","d","w"),
                ("w","w","w","w","w","w","w","d","d","w"),
                ("w","w","a","a","a","w","w","d","d","w")
            )
            return feedback_map[r][c]

        if feedback_policy_type == "heuristic_action_eval":
            """
            Attempts to model probability of giving action instruction
            according to Chi and Malle 2022, Figure 3,
            where ``appropriateness'' is a function of goal success
            and violations.

            General comments:
            (1) we'll first start out with some simple heuristics and
                then see if using these improves performance over random
                mixtures.
            (2) eventually the heuristics need to be informed from real
                human experiments data

            Parameters: Stochastic Goal Success, Stochastic Violations

            Heuristic 1: Based only on a goal success
                (a) threshold
                (b) linear
                (c) function
            Heuristic 2: Based only on violations (likely to fail)?
            Hueristic 3: Based on both

            Boundary Conditions:
                (1) 100% Goal Success, 0% violations = 0% action
                (2) 0% Goal Success, 100% violations = 100% action?
                (3) 100% goal success, >0% violations = >0% action?
                (4) 0% Goal Success, 0% Violations = 100% action
            """
            def heuristic_1(goal_success, violations):
                """
                 (V) ^
                    1|    1
                     |
                     |____>(G)
                    1     0

                Need a nonlinear surface fitting to
                (0,0,1), (0,1,1), (1,0,0), (1,1,1)
                Can be plane with points (0,0,1), (1,0,0), (1,1,1), with z-clipped
                between 0 and 1
                Ax + By + Cz = D
                C = D = k (0,0,1)
                A = D = C = k(1,0,0)
                A + B + C = D
                k + B + k = k, then B = -k
                Let k  = 1
                x - y + z = 1
                0 - 0 + 1 = 1
                1 - 0 + 0 = 1
                1 - 1 + 1 = 1
                goal_success - violations + prob_of_action_feedback = 1
                prob of action feedback = clip(1 - goal_sucess + violations)
                """
                return np.clip(1.0 - goal_success + violations, 0, 1)
            def heuristic_2(goal_success, violations):
                """
                 (V) ^
                    1| 1  0
                    1| 0  0
                    0|____>(G)
                          0
                Use (0, 0.25, 1), (0.25, 1, 1), (0.25,0.25, 0)
                Ax+By+Cz=1
                0.25B + C = 1
                0.25A + B + C = 1 =>  0.25A + 0.75B = 0 => A + 3B = 0 => A = -3B
                0.25A + 0.25B = 1 => A + B = 4 => -2B = 4, B = -2, A = 6, C = 1.5

                6x - 2y + 1.5z = 1
                """
                return np.clip(1.0 - 6*goal_success + 2*violations, 0, 1)

            def heuristic(goal_success, violations):
                #return heuristic_1(goal_success, violations)
                return heuristic_2(goal_success, violations)


            ## 1. Evaluate Goal Success and Violations Using stochastic metrics
            agent_policy, _, _ = agent_cg.forward(as_numpy=True)
            gs = self.stochastic_goal_success(agent_policy)
            pv = self.stochastic_policy_violations(agent_policy, violation_set)
            print(f"Eval Stoch. Goal Success: {gs}")
            print(f"Eval Stoch. Violations:   {pv}")
            ## 2. Compute probability from heuristic
            p = heuristic(gs, pv)
            print(f"Heur. 1 (Prob of A FB):   {p}")
            return self.feedback_source(action, r, c,
                    "mixed", agent_cg,
                    ["action_path_cost", "affine_policy_evaluation"],
                    p, violation_set)

        if feedback_policy_type == "action_path_cost":
            """
            Computes optimal action using the strategy in r1_evaluative
            """
            if self.need_simulated_evaluative_feedback or self._problem is None:
                goal_state = tuple(np.argwhere(self.world == 4)[0])
                prob = GridWorldProblem(self, goal_state, goal_state, violation_set)
                search_problem = GridWorldSearch(prob)
                costs, paths = search_problem.cost_to_goal()
                self.need_simulated_evaluative_feedback = False
                ## NOTE: These are the base costs from each point in the grid to the goal IF taking the shortest
                ## path. When decision making, need to add the cost of transitioning from the current state
                ## TO one of these cells to determine the actual cost of taking an action
                ## Example for world 1:
                ##   [[ 13.  12.  13.   8.   7.   6.   5.   4.   1.   0.]
                ##    [ 12.  11.  10.   7.   6.   5.   4.   3.   2.   1.]
                ##    [ 11.  10.   9.   8.   7.   6.   5.   4.   3.   2.]
                ##    [ 12.  11.  10.   9.   8.   7.   6.   5.   4.   3.]
                ##    [ 13.  12.  13.  14.  15. 107. 106.   6.   5.   4.]]
                ##
                self._costs = costs
                self._paths = paths
                prob.set_forward(forward=True)
                self._problem = prob
                self.idx2act = ["w","d","s","a","stay"]
                print(f"Entered simulation")

            decision_costs = np.zeros(len(self.actions))
            for idx in range(len(self.actions)):
                a = self.actions[idx]
                next_state = self._problem.transition((r,c), a)
                edge_cost = self._problem.transition_cost((r,c), a, next_state)
                decision_costs[idx] = self._costs[next_state] + edge_cost

            mapped_advantage = (decision_costs.argsort().argsort() - 2)*(-1)
            a_idx = np.argmax(mapped_advantage)
            return self.idx2act[a_idx]

        if feedback_policy_type == "r1_evaluative":
            """
            A very rough automated approximation to researcher 1's evaluative feedback strategy.
            This is based on ranking of path-cost, and then making adjustments for violations and other
            undesirable behaviors.

            This first solves a for optimal-cost path via uniform-cost-search for every position in the grid
            world to the goal
            """
            ## i.e. researcher 1's evaluative feedback teaching strategy:
            ## Whenever a violation is made, the feedback must be negative. Feedback is a scalar that is ranked
            ## and given such that the action brings the agent closer to the goal (i.e. along a shortest path)
            ## is considered the best.
            if self.need_simulated_evaluative_feedback or self._problem is None:
                goal_state = tuple(np.argwhere(self.world == 4)[0])
                prob = GridWorldProblem(self, goal_state, goal_state, violation_set)
                search_problem = GridWorldSearch(prob)
                costs, paths = search_problem.cost_to_goal()
                self.need_simulated_evaluative_feedback = False
                ## NOTE: These are the base costs from each point in the grid to the goal IF taking the shortest
                ## path. When decision making, need to add the cost of transitioning from the current state
                ## TO one of these cells to determine the actual cost of taking an action
                ## Example for world 1:
                ##   [[ 13.  12.  13.   8.   7.   6.   5.   4.   1.   0.]
                ##    [ 12.  11.  10.   7.   6.   5.   4.   3.   2.   1.]
                ##    [ 11.  10.   9.   8.   7.   6.   5.   4.   3.   2.]
                ##    [ 12.  11.  10.   9.   8.   7.   6.   5.   4.   3.]
                ##    [ 13.  12.  13.  14.  15. 107. 106.   6.   5.   4.]]
                ##
                self._costs = costs
                self._paths = paths
                prob.set_forward(forward=True)
                self._problem = prob
                self.idx2act = ["w","d","s","a","stay"]

            ## NOTE: Feedback is "advantage-based" in this scenario in that the (a,s') pairs are ranked according to
            ## the lowest cost of s'.
            ## The possible actions are given by self.env.actions
            ## NOTE: Decision costs are computed in the following way:
            ##   The agent is currently at (r,c) and will take an action a that takes it to (r',c')
            ##   Ranking of the action is determined by self._costs[(r',c')] + transition_cost((r,c), a, (r',c'))
            decision_costs = np.zeros(len(self.actions))
            for idx in range(len(self.actions)):
                a = self.actions[idx]
                next_state = self._problem.transition((r,c), a)
                edge_cost = self._problem.transition_cost((r,c), a, next_state)
                decision_costs[idx] = self._costs[next_state] + edge_cost
            ## NOTE: argsort does highest value = largest index; lowest value = lowest index
            ## In the lowest cost case, we want the highest index to have highest advantage
            ## For example: 
            ## If decision costs is : [101.   1.   3. 104. 101.]
            ## then the argsort is [1, 2, 0, 4, 3]  ** See the np.argsort() documentation
            ## To get the rank, do .argsort() again
            mapped_advantage = (decision_costs.argsort().argsort() - 2)*(-1)
            ## Addtional adjustments on "0" advantage cases:
            ## "Staying Put" has ranking of 0, then it should get -2 advantage
            ## "Violating action" has ranking of 0, give it -2
            zero_idx = np.where(mapped_advantage == 0)[0][0]
            ## Give staying put a negative value
            if mapped_advantage[4] == 0:
                mapped_advantage[4] -= 2
                if mapped_advantage[4] < -2:
                    mapped_advantage[4] = -2
            ## Transitions to a violating state should be negative
            if decision_costs[zero_idx] > 100:
                mapped_advantage[zero_idx] = -2
            ## Transitions that yield a larger min cost than current should be negative
            if decision_costs[zero_idx] < 100:
                ## If the action brings it to a state where the min cost is greater than the current state,
                ## then make sure that gets a negative value
                current_cost = self._costs[(r,c)]
                other_cost = self._costs[self._problem.transition((r,c), self.actions[zero_idx])]
                if other_cost > current_cost:
                    mapped_advantage[zero_idx] = -2
            print(decision_costs)
            print(mapped_advantage)
            ## Fix for giving 0 advantage to "staying put"
            #if action == 4 and mapped_advantage[action] == 0:
            #    return str(-2)
            return str(mapped_advantage[action])

        if feedback_policy_type == "ranked_evaluative":
            """
            "Ranks" the advantage values computed from a ground truth reward function
            i.e. runs "raw_evaluative" to get the advantage values, and then maps them onto
            a {-2, -1, 0, 1, 2} range
            This version is independent of the agent's policy
            """
            self.__compute_ground_truth()
            ## Consider just advantage mapped onto -2, -1, +0, +1, +2 via ranking.
            ## NOTE: Maps to -2, -1, +0, +1, +2 evaluative ranking that we have been using for experiments.
            ## If we don't want to do this, then just return the raw advantage
            ## To return raw advantage, set map_to_ranked_advantage_scale to False
            mapped_advantage = self.sim_advantage.argsort().argsort() - 2
            #print(mapped_advantage)
            #print(f"r: {r}, c: {c}, nrows: {nrows}, action: {action}")
            return str(mapped_advantage[r*nrows + c, action])
       
        if feedback_policy_type == "policy_evaluation":
            """
            The teacher evaluates an estimate of the agent's policy in order to provide feedback.
            For now, we provide the agent's exact policy, and evaluate it against a ground-truth
            reward function.  Returns raw advantage values.

            This evaluates the agent's policy against a ground truth reward function (hence making
            the feedback under this strategy policy dependent, inline with COACH).
            """
            agent_pi, agent_Q, agent_V = agent_cg.forward(as_numpy=True)
            self.__compute_ground_truth()
            A, Q, V = self.policy_evaluation(self.sim_cg.current_reward_estimate(), agent_pi)
            raw_advantage = A[r*nrows+c][action]
            print(f"Agent Policy: {agent_pi[r*nrows+c]}")
            print(f"A: {A[r*nrows+c]}")
            print(f"Q: {Q[r*nrows+c]}")
            print(f"V: {V[r*nrows+c]}")
            return str(raw_advantage)
        
        if feedback_policy_type == "affine_policy_evaluation":
            """
            Same as ``policy evaluation'', except performs an affine transformation of the
            advantage values.
            """
            scale = 4
            agent_pi, agent_Q, agent_V = agent_cg.forward(as_numpy=True)
            self.__compute_ground_truth()
            A, Q, V = self.policy_evaluation(self.sim_cg.current_reward_estimate(), agent_pi)
            sorted_adv = np.sort(A[r*nrows+c])

            ## If multiple values are the same as the max, then find the first one that is
            ## outside the tolerance range.
            A_max = sorted_adv[-1]
            A_min = sorted_adv[0]

            ## If the min and the max are the same, then everything has the same advantage.
            affine_A = A[r*nrows+c] ## Don't perform any transformation
            if A_max - A_min >= 1e-12:
                start_idx = -2
                end_idx = -5
                idx = start_idx
                A_sec = sorted_adv[idx]
                while A_max - A_sec < 1e-12 and idx > end_idx:
                    idx = idx - 1
                    A_sec = sorted_adv[idx]
                affine_A = (A[r*nrows+c] - (A_max + A_sec)*0.5)*scale/(A_max - A_min + 1e-24)
            
            raw_advantage = affine_A[action]
            print(f"Agent Policy: {agent_pi[r*nrows+c]}")
            print(f"A: {A[r*nrows+c]}")
            print(f"affine_A: {affine_A}")
            print(f"Q: {Q[r*nrows+c]}")
            print(f"V: {V[r*nrows+c]}")
            return str(raw_advantage)

        if feedback_policy_type == "ranked_policy_evaluation":
            """
            The teacher evaluates an estimate of the agent's policy in order to provide feedback.
            For now, we provide the agent's exact policy, and evaluate it against a ground-truth
            reward function.  Returns raw advantage values.

            This evaluates the agent's policy against a ground truth reward function (hence making
            the feedback under this strategy policy dependent, inline with COACH).
            """
            if action == 4:
                return str(-2) ## Discourage "staying"
            agent_pi, agent_Q, agent_V = agent_cg.forward(as_numpy=True)
            self.__compute_ground_truth()
            A, Q, V = self.policy_evaluation(self.sim_cg.current_reward_estimate(), agent_pi)
            raw_advantage = A[r*nrows+c][action]
            print(f"Agent Policy: {agent_pi[r*nrows+c]}")
            print(f"A: {A[r*nrows+c]}")
            print(f"Q: {Q[r*nrows+c]}")
            print(f"V: {V[r*nrows+c]}")
            ranked_advantage = A.argsort().argsort() - 2
            ## Set 0 to -1
            ranked_advantage[ranked_advantage == 0] = -1
            return str(ranked_advantage[r*nrows+c, action])
 
        if feedback_policy_type == "raw_evaluative":
            """
            Uses the raw advantage values computed from a ground truth reward function
            This version is independent of the agent's policy
            """
            self.__compute_ground_truth()
            ## TODO: Support raw advantage values (for arbitrary scaling)
            ## Alternatively, we should also be able to handle direct advantage feedback
            ## Scalar only
            ## Return advantage value
            ## Assumes a reward function that yields correct optimal behavior
            ## However, what's interesting that is that depending on the scale,
            ## the advantage values will be different! So maybe normalize the values.
            raw_advantage = self.sim_advantage[r*nrows+c][action]
            print(f"Advantage: {self.sim_advantage[r*nrows+c]}")
            print(f"Policy: {self.policy[r*nrows+c]}")
            print(f"QVals: {self.Qvals[r*nrows+c]}")
            print(f"Vvals: {self.Vvals[r*nrows+c]}")
            #print(self.sim_advantage)
            ## NOTE: This is not yet supported
            #raise ValueError("Raw advantage support not yet implemented.")
            return str(raw_advantage)

        if feedback_policy_type == "mixed":
            """
            This is supposed to represent a placeholder for some mix of action and evaluative feedback
            """
            ### Create a static mixture of the action and evaluative feedback
            ### Create a dynamic mixture of action and evaluative feedback (distribution changes over time)
            ### Create a mixture dependent on the history of the state, action, feedback encountered
            if self.rng.uniform() < mixed_percent:
                return self.feedback_source(action, r, c, mixed_strat[0], agent_cg, mixed_strat, mixed_percent, violation_set)
            else:
                return self.feedback_source(action, r, c, mixed_strat[1], agent_cg, mixed_strat, mixed_percent, violation_set)

    def policy_evaluation(self, reward_func, policy):
        """
        Evaluates a policy under a ground truth reward function.
        reward_func = a feature-based reward function to evaluate the policy against
            numpy_array or pytorch.tensor
        policy = a policy to evaluate against the reward (i.e. find V and Q-values under this policy)
            numpy_array or pytorch.tensor
        """
        ## Solve system of equations to determine V^pi, or just iterate Bellman equation
        ## policy = |S| x |A|
        stochastic_pi = torch.tensor(policy, dtype=self.dtype, requires_grad=False)
        ## Transition matrix is |A|x|S|x|S'|
        if self.transition_matrix is None:
            self.transition_matrix = self.get_mattrans()
        if self.matrix_map is None:
            self.matrix_map = self.get_matmap()

        ## reward func is |F|-length vector. we want to compute R(s,a), which is the expected reward you get
        ## for taking action (a) in state (s) (and leading you into state s'). We can construct this from the
        ## transition matrix: R(s,a) = T(a,s,s')R(s'), then sum reduce across the S' dimension (i.e. take inner product)
        ## This will result in a |A|x|S| reward matrix
        r_s = self.__compute_reward_s(reward_func)
        r_sa = self.__compute_reward_sa(r_s)
        ## one step expected return: |S| element vector:
        ## one_step = \sum_a \pi(a|s)r(s,a)
        one_step = (stochastic_pi*(r_sa.T)).sum(-1)
        
        ## Initialize the Value function to this
        V = r_s.detach().clone()

        g = 0.99 ## gamma
        for _ in range(100):
            ## Bellman update
            ## V^\pi(s) = \sum_a \pi(a|s)r(s,a) + g\sum_a \pi(a|s) \sum_{s'} T(s'|s,a)V^\pi(s')
            V = one_step + g*((stochastic_pi*(torch.matmul(self.transition_matrix, V).T)).sum(-1))

        ## We can compute Q^\pi from V^\pi, T
        ## Q^\pi(s,a) = r(s,a) + g\sum_{s'} T(s'|s,a)V^\pi(s')
        ## Do transpose to get a |S|x|A| matrix
        Q = (r_sa + g*torch.matmul(self.transition_matrix, V)).T
        A = self.__compute_advantage(Q, V.detach().clone())
        return A.numpy(), Q.numpy(), V.numpy()

    def __compute_reward_sa(self, rs):
        """
        Computes a R(s,a) reward function using R(s') and T(a,s,s')
        """
        if self.transition_matrix is None:
            self.transition_matrix = self.get_mattrans()
        ## If  |A|x|S|x|S'| x |S'|, then do torch.matmul(T, R)
        ## If  |A|x|S|x|S'| x |S'|x1, then do torch.matmul(T, R.squeeze()) or torch.matmul(T, R).sum(-1) or torch,matmul(T,R).squeeze()
        r_sa = torch.matmul(self.transition_matrix, rs)  ## should be |A| x |S|
        return r_sa

    def __compute_reward_s(self, reward_func):
        """
        Computes a R(s) reward function using the feature-based reward function and a matrix map
        """
        if self.matrix_map is None:
            self.matrix_map = self.get_matmap()
        rk = torch.tensor(reward_func, dtype=self.dtype, requires_grad=False)
        new_rk = rk.unsqueeze(0)
        new_rk = new_rk.unsqueeze(0)
        nrows, ncols, ncategories = self.matrix_map.shape
        new_rk = new_rk.expand(nrows, ncols, ncategories)
        ## Dot product to obtain the reward function applied to the matrix map
        rfk = torch.mul(self.matrix_map, new_rk)
        rfk = rfk.sum(axis=-1)
        rffk = rfk.view(nrows*ncols) ## 1D view of the 2D grid
        #initialize the value function to be the reward function, required_grad should be false
        return rffk

    def __compute_advantage(self, Q, V):
        """
        Computes A = Q-V

        Expects Q = |S|x|A| (2-D), V = |S| (1-D)
        """
        V = V.unsqueeze(0).T.expand(Q.shape)
        ## Advantage
        A = Q - V ## 2D nrows*ncols x nacts
        return A

    def __compute_ground_truth(self):
        """
        This computes an optimal policy from a feature-based ground truth reward function.
        """
        if self.sim_advantage is None or self.need_simulated_evaluative_feedback:
            ## We haven't computed the simulated advantage values yet,
            ## so compute those values using a prespecified reward function
            self.sim_cg = ComputationGraph(self)
            ## Specify a reward function to use
            ## default_0
            ## DEFAULT_REWARD = [0, -1, -1, -1,  1]
            ## default_1
            ## DEFAULT_REWARD = [1.3605843, -1.1135638, -0.62023115, -1.9482319, 2.837054]
            ## default_2
            ## DEFAULT_REWARD = [ 0.6487329, -2.6717327, -1.280116, 0.519958, 2.7245345]
            ## default_3 (from supervised action-only version, probably most accurate?)
            #DEFAULT_REWARD = [ 0.5631, -0.4677, -0.4628, -0.7351,  1.1025]
            ## default_4
            DEFAULT_REWARD = [0, -1, -1, -10,  100]
            reward = np.array(DEFAULT_REWARD) - DEFAULT_REWARD[0] - 0.01
            reward = reward / np.linalg.norm(reward)
            self.sim_cg.set_reward_estimate(reward)
            pi, Q, V = self.sim_cg.forward()
            ## Q is (nrows*ncols) x (nacts) (2D)
            ## V is (nrows*ncols) (1D)
            V = V.unsqueeze(0).T.expand(Q.shape)
            ## Advantage
            A = (Q - V).cpu().detach().numpy() ## 2D nrows*ncols x nacts
            self.need_simulated_evaluative_feedback = False
            self.sim_advantage = A
            self.policy = pi
            self.Qvals = Q
            self.Vvals = V
            #self.sim_cg = None

    def random_start_state(self):
        """
        Select a random start state (chooses a random (r,c) that is a 0 colored square)
        This is now a generator.
        """
        indices = np.flatnonzero(self.world == 0)
        total = len(indices)
        _, ncols = self.world.shape
        n = 0
        while True:
            if n == 0:
                np.random.shuffle(indices)
            idx = indices[n]
            r = idx // ncols
            c = idx %  ncols
            yield r, c
            n = (n + 1) % total

    def get_violation_matrix(self, violation_set):
        if self.violation_matrix is None:
            self.violation_matrix = self.compute_violation_matrix(violation_set)
        return self.violation_matrix

    def simulate_markov_chain(self, pi, steps, verbose=True):
        """
        G: S x A -> S is the environment transition function (it returns
        the state you will be in when taking action a in state s)

        T is the Markov Chain transition matrix, T: S x S -> [0, 1]
        T(s'|s) = \sum_a \pi(a|s)T(s'|s,a)

        States are index by r*nrows + c, with s = (r,c)

        Strategy: We construct the transition matrix T, for an arbitrary
        policy
        """
        nrows, ncols = self.world.shape
        state_size = nrows*ncols
        
        assert np.all(pi<=1)
        assert np.all(0<=pi)
        assert not np.isnan(pi).any()
        assert np.all(np.sum(pi, axis=1) <= 1.00001)

        pi[:,:] += 1e-24
        pi = pi / np.sum(pi, axis=1, keepdims=True)
        assert np.all(pi<=1)
        assert np.all(0<=pi)
        assert not np.isnan(pi).any()
        assert np.all(np.sum(pi, axis=1) <= 1.00001)

        ## Initialization
        T = np.zeros((state_size, state_size), dtype=float)
        G = [] ## Track the goal state indices here
        gfeature = self.get_goal_state()
        ## Populate transition matrix, assuming deterministic environment
        ## transitions, and probabilistic policy actions
        for r in range(nrows):
            for c in range(ncols):
                for a in range(len(self.actions)):
                    n_r, n_c = self.transition(r, c, a)
                    i =   r * nrows +   c   ## Current state
                    j = n_r * nrows + n_c   ## Next state
                    T[i,j] = T[i,j] + pi[i,a]
                if self.world[r,c] == gfeature:
                    G.append(r*nrows+c) ## Index of goal state

        ## Make sure to normalize
        T[:,:] += 1e-24 ## Take care of any zero rows, if there are any
        T = T / np.sum(T, axis=1, keepdims=True)

        assert np.all(T<=1.0001)
        assert np.all(0<=T)
        assert not np.isnan(T).any()
        assert np.all(np.sum(T, axis=1)<= 1.0001)

        ## Set the goal states to be absorbing states
        for g in G:
            T[g, :] = 0.0
            T[g,g] = 1
        if verbose:
            print(f"T: {T}")

        ## Set uniform initial distribution, except in goal states:
        num_nongoal = state_size - len(G)
        d = np.ones(state_size, dtype=float)
        d /= num_nongoal
        for g in G:
            d[g] = 0.0
        ## Normalize
        d = d / np.sum(d)
        if verbose:
            print(f"d: {d}")

        ## Simulate the Markov Chain for N steps
        for n in range(steps):
            d = np.matmul(d, T)
            d = d / np.sum(d)
        if verbose:
            print(f"dN: {d}")
        return d, G

    def stochastic_policy_violations(self, policy, violation_set):
        """
        Computes policy violations
        """
        nrows, ncols = self.world.shape
        state_size = nrows*ncols
        V = self.get_violation_matrix(violation_set)
        ## Take the element-wise product, then add up all the probability mass
        ## and then divide by the state_size
        exp_violations = np.sum(V*policy) / state_size
        return exp_violations

    def stochastic_goal_success(self, policy):
        """
        Computes goal success

        N is the total state size (as an integer)
        policy is a np.array representing the policy
        """
        nrows, ncols = self.world.shape
        N = nrows*ncols
        distribution, goal_indices = self.simulate_markov_chain(policy, N, False)
        goal_success = 0.0
        for g in goal_indices:
            goal_success = goal_success + distribution[g]
        return goal_success

    def compute_violation_matrix(self, violation_set):
        """
        Computes the violations matrix. The form of the matrix is:
        |S| x |A|. Here, the |S| dimension is indexed by r*nrows+c for
        state s = (r,c)
        Similarly, the indexing for |A| correspond to the default indices
        that are used for the actions vector.

        Each entry of the violation matrix will be filled with either
        a 0 or a 1. If the entry is 1, then this means taking action a
        while in state s counts as a violating action.

        We define computing the violations as follows:
        Let E represent the environment transition for taking action a
        in state s, that is E: S x A -> S
        Additionally, let S_V \subset S be the set of violating features
        Let \phi be a function that returns the feature of a state s
        Let V represent the violations matrix

        For each s for each a:
        Let s' = E(s,a)
        If s' == s or \phi(s') \in S_V
            V(s,a) = 1
        Else:
            V(s,a) = 0
        """
        vfeature = set(violation_set)
        print(f"Violation Set: {vfeature}")
        nrows, ncols = self.world.shape
        V = np.zeros((nrows*ncols, len(self.actions)), dtype=int)
        for r in range(nrows):
            for c in range(ncols):
                for a in range(len(self.actions)):
                    n_r, n_c = self.transition(r, c, a)
                    V[r*nrows+c, a] = int(((n_r, n_c) == (r, c)) or self.world[n_r, n_c] in vfeature)
        return V


    def get_termination_states(self):
        """
        Return the location of the termination state
        """
        pass

    def get_goal_state(self):
        """
        Returns the color that is considered a goal
        """
        return 4

    def get_start_state(self):
        return copy.copy(self.state_start)

    ## Helpers for goal success
    def transition(self, r, c, a):
        nrows, ncols = self.world.shape
        r_next = r + self.actions[a][0]
        c_next = c + self.actions[a][1]
        r_next = min(r_next, nrows-1)
        r_next = max(r_next, 0)
        c_next = min(c_next, ncols-1)
        c_next = max(c_next, 0)
        return r_next, c_next

    def step(self, r, c, action):
        """
        Transitions r,c based on action
        """
        rr, cc = self.actions[action]
        return (r + rr, c + cc)

    def get_single_timestep_action(self):
        """
        Retrieves a single discrete timestep action from a human
        """
        k = input("Action: ")
        if k not in self.action_feedback_map:
            return -1
        else:
            return self.action_feedback_map[k]

    def acquire_human_demonstration(self, max_length=15):
        """
        Acquires a trajectory of actions from a human
        of length of at least 1, starting from an initial state
        """
        cur_state = self.get_start_state()
        self.visualize_environment(cur_state[0],cur_state[1])
        action = self.get_single_timestep_action()
        action_sequence = [action]
        ## Create a clone of the start state and allow the human to demonstrate from there
        ## Update the current state
        cur_state = [a+b for a,b in zip(cur_state, self.act_map[action])]
        print(f"Action({action})={self.act_name[action]}")

        for step in range(max_length):
            self.visualize_environment(cur_state[0],cur_state[1])
            action = self.get_single_timestep_action()
            if action == -1:
                break
            action_sequence.append(action)
            cur_state = [a+b for a,b in zip(cur_state, self.act_map[action])]
            print(f"Action({action})={self.act_name[action]}")

        trajcoords = reduce((lambda seq, a: seq + [[seq[len(seq)-1][0] + self.actions[a][0], seq[len(seq)-1][1] + self.actions[a][1]]]), action_sequence, [self.state_start])
        return action_sequence, trajcoords

    def get_matmap(self):
        """
        Return binarized form of a grid map:
        (rows, cols, category) -- true false for the category
        """
        r,c = self.world.shape
        shape = (r,c,len(self.categories))
        matmap = np.zeros(shape)
        for k in self.categories:
            matmap[:,:,k] = 0 + (self.world == k)
        return torch.tensor(matmap, dtype=self.dtype, requires_grad=False)

    def get_mattrans(self):
        """
        Return transition matrix
        """
        def clip(v,min_v,max_v):
            if v < min_v: v = min_v
            if v > max_v-1: v = max_v-1
            return(v)
        nrows,ncols = self.world.shape
        nacts = len(self.actions)
        mattrans = np.zeros((nacts, nrows*ncols, nrows*ncols))
        for acti in range(nacts):
            act = self.actions[acti]
            for i1 in range(nrows):
                for j1 in range(ncols):
                    inext = clip(i1 + act[0],0,nrows)
                    jnext = clip(j1 + act[1],0,ncols)
                    for i2 in range(nrows):
                        for j2 in range(ncols):
                            mattrans[acti,i1*ncols+j1,i2*ncols+j2] = 0+((i2 == inext) and (j2 == jnext))
        return torch.tensor(mattrans, dtype=self.dtype, requires_grad=False)

    def visualize_environment(self, robox, roboy):
        print("===================================")
        for r in range(len(self.world)):
            rowstr = ""
            for c in range(len(self.world[r])):
                if r==robox and c==roboy:
                    rowstr += "R"
                else:            
                    rowstr += str(self.world[r][c])
            print(rowstr)
