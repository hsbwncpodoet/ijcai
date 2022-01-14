from queue import Queue, LifoQueue, PriorityQueue
import numpy as np

class GridWorldProblem:
    """
    states are represented as (r,c) tuples
    """
    def __init__(self, env, initial_state, goal_state, violating_set):
        self.env = env
        self.init_state = initial_state
        self.goal_state = goal_state
        self.forward = True
        self.violating_set = violating_set

    def set_forward(self, forward=True):
        """
        Sets the transition directions, so that we can do forward or backward (reverse) search
        By default, everything is setup as a forward search
        """
        self.forward = forward

    def initial_state(self):
        return self.init_state

    def is_goal(self, state):
        return state == self.goal_state

    def actions(self, state):
        r, c = state
        nrows, ncols = self.env.world.shape
        if (0 < r and r < nrows-1) and (0 < c and c < ncols-1):
            return self.env.actions

        ## Otherwise, just return the actions that don't bring the agent out of bounds
        filtered_actions = [x for x in self.env.actions]
        if self.forward:
            if r == 0:
                filtered_actions.remove((-1,0))
            if r == nrows-1:
                filtered_actions.remove((1,0))
            if c == 0:
                filtered_actions.remove((0,-1))
            if c == ncols-1:
                filtered_actions.remove((0,1))
            return filtered_actions
        else:
            ## Do backward filtering
            if r == 0:
                filtered_actions.remove((1,0))
            if r == nrows-1:
                filtered_actions.remove((-1,0))
            if c == 0:
                filtered_actions.remove((0,1))
            if c == ncols-1:
                filtered_actions.remove((0,-1))
            return filtered_actions

    def transition(self, state, action):
        if self.forward:
            return self.__forward_transition(state, action)
        else:
            return self.__backward_transition(state, action)
    
    def transition_cost(self, state, action, next_state):
        if self.forward:
            return self.__forward_transition_cost(state, action, next_state)
        else:
            return self.__backward_transition_cost(state, action, next_state)

    def __clip(self, v, min_v, max_v):
        if v <= max_v and v >= min_v:
            return v
        else:
            if v > max_v:
                return max_v
            if v < min_v:
                return min_v

    def __forward_transition(self, state, action):
        """
        Forward transition: the transition is starting in state s and taking action a
        what is the next state that you end up in?
        """
        nrows, ncols = self.env.world.shape
        dr, dc = action
        r, c = state
        R = r+dr
        C = c+dc
        ## Clip to stay in bounds
        R = self.__clip(R, 0, nrows-1)
        C = self.__clip(C, 0, ncols-1)
        return (R, C)
    
    def __backward_transition(self, state, action):
        """
        Backward transition: the backward transition is in which prev_state s' will taking action a
        lead to the current state s?

        We are in current state s = (r,c), taking action a = (dr, dc) will lead us from which state s' to s?
        (R, C) + (dr, dc) = (r, c) => (R,C) = (r,c) - (dr,dc)
        """
        nrows, ncols = self.env.world.shape
        dr, dc = action
        r, c = state
        R = r - dr
        C = c - dc
        #print(f"    Testing: {(R,C)} = {(r,c)} - {(dr, dc)}")
        ## Clip to stay in bounds
        R = self.__clip(R, 0, nrows-1)
        C = self.__clip(C, 0, ncols-1)
        #print(f"    Clip: {(R,C)} = clipped")
        return (R, C)

    def __forward_transition_cost(self, state, action, next_state):
        """
        What is the cost of transitioning FROM state s TO next state s' via action a?
        """
        base_cost_per_action = 1.0
        multiplier = 1.0
        r, c = next_state
        feature = self.env.world[r,c]
        ## If the action takes the agent to a violating state, assume very high cost
        ## otherwise, if action takes the agent to a non-violating state, cost is just 1
        if feature in self.violating_set:
            multiplier = 100.0
        return multiplier*base_cost_per_action
    
    def __backward_transition_cost(self, state, action, next_state):
        """
        What is the cost of transitioning TO state s FROM next state s' via action a?
        """
        base_cost_per_action = 1.0
        multiplier = 1.0
        r, c = state
        feature = self.env.world[r,c]
        ## If the action takes the agent to a violating state, assume very high cost
        ## otherwise, if action takes the agent to a non-violating state, cost is just 1
        if feature in self.violating_set:
            multiplier = 100.0
        return multiplier*base_cost_per_action

    def get_successors():
        pass


class GridWorldSearch:
    """
    Allows a search algorithm to be run on a GridWorld-based problem
    """
    def __init__(self, problem):
        self.problem = problem

    def uniform_cost_search(self):
        problem = self.problem

        node = Node(problem.initial_state())
        frontier = PriorityQueue()
        frontier_dict = dict()
        visited = dict()

        frontier.put(node)
        #visited[problem.initial_state()] = node
        frontier_dict[problem.initial_state()] = node

        print(f"Added {node.state} to the frontier")

        while True:
            if frontier.empty():
                print(f"Frontier is empty")
                return [], None ## Failure
            node = frontier.get(block=False)
            _ = frontier_dict.pop(node.state, None)
            print(f"Removed {node.state} from the frontier")
            if problem.is_goal(node.state):
                print(f"{node.state} is the goal state")
                return node.path(visited), visited
            print(f"Adding {node.state} to visited")
            visited[node.state] = node

            for action in problem.actions(node.state):
                print(f"  Considering action {action}")
                child = node.child_node(problem, action)
                print(f"  Taking {action} from {node.state} to {child.state}")
                if child.state not in visited and child.state not in frontier_dict:
                    print(f"  Adding to frontier: child={child}")
                    frontier.put(child)
                    frontier_dict[child.state] = child
                else:
                    if  child.state in frontier_dict and frontier_dict[child.state].totalcost > child.totalcost:
                        print(f"  Lower cost option found: updating {child}")
                        frontier_dict[child.state].pathcost = child.pathcost
                        frontier_dict[child.state].action = child.action
                        frontier_dict[child.state].parent = child.parent
                        frontier_dict[child.state].edge_cost = child.edge_cost
                        frontier_dict[child.state].heuristic_cost = child.heuristic_cost
                        frontier_dict[child.state].totalcost = child.totalcost
                        ## Re-sort all the elements in the frontier priority queue
                        ## by re-inserting all the existing elements into the queue.
                        ## This is inefficient, but w/e
                        frontier = PriorityQueue()
                        for k, v in frontier_dict.items():
                            frontier.put(v)

    def cost_to_goal(self):
        """
        Find all shortest paths to the goal

        returns a dict of states -> [goal, ...., state]
        """
        nrows, ncols = self.problem.env.world.shape
        self.problem.set_forward(forward=True)
        costs = dict()
        paths = dict()
        cost_array = np.zeros((nrows, ncols))
        for r in range(nrows):
            for c in range(ncols):
                self.problem.init_state = (r,c)
                ## Note that the shortest path is stored in reverse
                print(f"Computing shortest path from {self.problem.initial_state()} to {(r,c)}")
                shortest_path, explored = self.uniform_cost_search()
                costs[(r,c)] = shortest_path[-1].totalcost
                paths[(r,c)] = shortest_path
                cost_array[r,c] = costs[(r,c)]
        print(cost_array)
        #exit()
        return costs, paths

class Node:
    """
    Implements Norvig Node class on page 74
    """

    def __init__(self, state, parent=None, action=None, pathcost=0, edgecost=0, heuristic_cost=0):
        self.state = state ## The state to store in this Node
        self.parent = parent ## The parent Node of this Node
        self.action = action ## The action taken to reach this node from the Parent
        self.pathcost = pathcost ## This is the cost-so-far to reach this node
        self.edge_cost = edgecost ## This is the edge cost from parent to this node
        self.heuristic_cost = heuristic_cost ## This is the estimate of the cost-to-go

        self.totalcost = pathcost + heuristic_cost ## The total cost of going from start to goal via this state

    def get_parent(self):
        return self.parent

    def __str__(self):
        """
        Returns a string representation of the Node:
        """
        parent = None
        if self.parent is not None:
            parent = self.parent.state
        return f"{self.state} <- {parent} via a={self.action} | cost={self.totalcost}"

    def child_node(self, problem, action):
        """
        Returns a child node of parent, given that action is taken
        """
        next_state = problem.transition(self.state, action)
        edge_cost = problem.transition_cost(self.state, action, next_state)

        return Node(
            state=next_state,
            parent=self,
            action=action,
            pathcost=self.pathcost + edge_cost,
            edgecost = edge_cost
        )

    def expand(self, problem, heur):
        """
        Returns the children of this node as a list of Nodes
        Each node in this last has the following costs computed:
        - Cost-so-far (cost of going from start to parent to this child node)
        - Edge-Cost (cost of transitioning from the parent to the child node)
        - Heuristic-cost (an estimate of the cost-to-go or remaining cost from the child to a goal node)
        """
        successor_costs = problem.get_successors(self.state)
        ## Cost is cost-so-far (g) + transition_cost (c) + cost-to-go (h)
        ## This gives the cost-so-far for the path to the child + the cost-to-go estimate
        return [
            Node(
                state=child, 
                parent=self,
                action=None,
                pathcost=self.pathcost + transition_cost,
                edgecost=transition_cost,
                heuristic_cost=heur(child)
            ) 
            for child, transition_cost in successor_costs.items()
        ]

    def path(self, visited):
        """
        Returns the path from the root to this node using a list []
        """
        print(f"Reconstructing shortest path...")
        print(f"Visited Set is {visited}")
        p = []
        node = self
        print(f"    {node}")
        stack = LifoQueue()
        while node is not None:
            stack.put(node)
            node = node.get_parent()
            print(f"    {node}")

        while not stack.empty():
            node = stack.get()
            p.append(node)
    
        print(f"{p}")

        return p

    ## For the priority queue, we need to define comparison functions
    def __lt__(self, other):
        return self.totalcost <  other.totalcost
    def __le__(self, other):
        return self.totalcost <= other.totalcost
    def __gt__(self, other):
        return self.totalcost >  other.totalcost
    def __ge__(self, other):
        return self.totalcost >= other.totalcost
    def __eq__(self, other):
        return self.totalcost == other.totalcost
    def __ne__(self, other):
        return self.totalcost != other.totalcost









