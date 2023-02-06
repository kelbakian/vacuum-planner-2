from __future__ import annotations
import heapq
import sys
import math
from dataclasses import dataclass, field
from copy import deepcopy
from a2 import StateRepresentation
from typing import Any

class Problem:
    """Abstract framework for problem formulation. Contains an initial state, 
    goal test function/goal states, available actions,transition model, and
    action cost function (see pg. 65 of txtbook)"""
    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the set of possible actions the agent can execute in the state."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the resultant state produced by executing the given action in the given state."""
        raise NotImplementedError
    
    #general goal_test, maybe make this NotImplemented too? Not sure atm
    def goal_test(self, state):
        """Return True if the state is a goal."""
        return state == self.goal

    def action_cost(self, c, state1, action, state2):
        """Return the cost to go from state1 to state2 through the action,
        given current total cost c."""
        return c + 1

#Subclass the general framework to formulate vacuum world which is a 2d grid world problem    
class VacuumWorld(Problem):
    """Defines the vacuum path planning problem for an agent in a 2d grid environment."""

    def __init__(self, initial, goal,world,heuristic):
        super().__init__(initial,goal)
        self.dimrow = world.dimrow
        self.dimcol = world.dimcol
        self.possible_actions = world.possible_actions
        self.walls = world.obstacles
        self.chargers = world.chargers
        self.fuel_capacity = world.fuel_capacity
        self.h = None if heuristic==None else getattr(self,heuristic)
        
        

    def actions(self, state):
        row, col = state.loc
        valid_actions = self.possible_actions[:]
        if self.fuel_capacity is not None and state.remaining_fuel <= 0:
            if (row,col) not in self.chargers:
                return []
            else:
                return ['R']
        if row == 0 or (row-1,col) in self.walls:
            valid_actions.remove('N')
        if row == self.dimrow or (row+1,col) in self.walls:
            valid_actions.remove('S')
        if col == self.dimcol or (row,col+1) in self.walls:
            valid_actions.remove('E')
        if col == 0 or (row,col-1) in self.walls:
            valid_actions.remove('W')
        if (row,col) not in state.dirt_list:
            valid_actions.remove('V')
        if self.fuel_capacity is not None and (row,col) not in self.chargers:
            valid_actions.remove('R')
    
        return valid_actions

    def result(self, state, action):
        row, col = state.loc
        dirt = deepcopy(state.dirt_list)
        fuel = None

        if action == 'V':
            if (row,col) in dirt:
                dirt.discard((row,col))
        elif action == 'N':
                row -= 1
        elif action == 'S':
                row += 1
        elif action == 'E':
               col += 1
        elif action == 'W':
                col -= 1
        elif action == 'R':
            if (row,col) in self.chargers:
                return StateRepresentation((row,col),dirt,self.fuel_capacity)
        
        if self.fuel_capacity is not None:
            fuel = state.remaining_fuel - 1
        return StateRepresentation((row,col),dirt,fuel)
    
    def goal_test(self, state):
        return state.dirt_list == self.goal
    
    #Add heuristic functions to the problem sublcass
    def h0(self,node):
        return 0

    def h1(self,node):
        """Manhattan distance heuristic |x1-x2| + |y1-y2|"""
        state = node.state
        x,y = state.loc
        dist = 0
        for goal in state.dirt_list:
            dist += abs(x-goal[0]) + abs(y-goal[1])
            x,y = goal[0],goal[1]
        return dist

    def h2(self,node):
        """Euclidean distance heuristic"""
        state = node.state
        x,y = state.loc
        dist = 0
        for goal in state.dirt_list:
            dist += math.sqrt((goal[1]-y)*(goal[1]-y) + (goal[0]-x)*(goal[0]-x))
            x,y = goal[0],goal[1]
        return dist


    def h3(self,node):
        """heuristic that incorporates vacuum's charge by using
        the # of times a vacuum would have had to recharge on its
        path to a goal state assuming no obstacles"""
        state = node.state
        x,y = state.loc
        cost = 0
        fuel = state.remaining_fuel
        for goal in state.dirt_list:
            dist = abs(x-goal[0]) + abs(y-goal[1])
            if dist >= fuel:
                cost += 1
            else:
                fuel -= dist
            x,y = goal[0],goal[1]
        return cost
        

# _____AGENT FRAMEWORK AND VACUUM AGENT SUBCLASS__________________________________________________

class ProblemSolvingAgent:
    """
    Abstract framework for a problem-solving agent - an agent that plans ahead before
    executing any actions.
    Defined by 4 phases: goal formulation, problem formulation, search, and execution.
    (see pg. 77 txtbook)
    """

    def __init__(self, initial_state=None,algo=None):
        self.state = initial_state
        self.algo = algo
        self.res = None


    def __call__(self):
        """Calling the agent should formulate a goal and problem, then search for the solution and return it.
        Specific implementation depends on the type of problem/agent/env. 
        (e.g. if world fully observed and deterministic, or if probabilistic, or state depends on percepts, etc.)"""
        raise NotImplementedError
        
    
    def update_state(self, state, percept):
        """Update agent's state given a percept. Used for online agents, not used when state-space is fully observable and deterministic."""
        raise NotImplementedError
    
    def formulate_goal(self, state=None):
        """Goal the agent will adopt (e.g. cleaning all dirty cells)."""
        raise NotImplementedError
   
    def formulate_problem(self, state, goal):
        """Description of the states and actions necessary to reach the goal.
        An abstract model of the relevent world parts.
        What fact(s) about the state of the world will change due to applying
        an action? E.G. current location, dirty cell"""
        raise NotImplementedError
    
    def solve(self, problem):
        """Agent simulates sequences of actions until it finds a path to its goal. 
        May have to simulate mutliple sequences that don't reach a goal
        first of course, but will find sol'n eventually or it DNE (assuming closed, finite env)"""
        raise NotImplementedError

class SimpleVacuumAgent(ProblemSolvingAgent):
    """path-finding vacuum agent in a deterministic, fully observable, environment."""
    
    def __init__(self,world,initial_state=None,algo=None,heuristic=None):
        super().__init__(initial_state,algo)
        self.world = world
        self.heuristic = heuristic

    def __call__(self):
        goal = self.formulate_goal(self.state)
        problem = self.formulate_problem(self.state, goal,self.heuristic)
        self.res = self.solve(problem)
        if not self.res:
            return None
        return self.res

    def formulate_goal(self,state):
        """Goal states are any in which every cell is clean"""
        return set()

    def formulate_problem(self, state, goal,heuristic):
        """Creates the vacuum world problem instance based on given input"""
        return VacuumWorld(state,goal,self.world,heuristic) 

    def solve(self, problem):
        """Agent creates plan/finds goal state using user-supplied algorithm choice."""
        if self.algo == "depth-first":
            return depth_first_search(problem)
        elif self.algo == "uniform-cost": 
            return uniform_cost_search(problem)
        elif self.algo == "depth-first-id":
            return iterative_deepening_search(problem)
        elif self.algo == "a-star":
            return a_star_search(problem)
        elif self.algo == "ida-star":
            return ida_star_search(problem)
        elif self.algo == "greedy":
            return greedy_search(problem)
        else:
            sys.exit("invalid search algorithm supplied")

@dataclass()
class Node:
    """A wrapper for the state. Used to build the state-space search tree"""
    state: StateRepresentation = None
    parent: Node = None
    action: str = None
    g: int = 0
    depth: int = 0
    
    def __post_init__(self):
        if self.parent:
            self.depth = self.parent.depth + 1

    #For easier debugging return node in string format
    def __repr__(self):
        return "Node {}".format(self.state,self.parent,self.action,self.g,self.depth)
    
    #Overload necessary operators s.t. we avoid duplicate states/cycles and the priority queue can order nodes
    def __lt__(self, node):
        return self.g < node.g
    
    #Node equivalency based on states
    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def expand(self,problem):
        """Generator for the given node's child nodes"""
        s = self.state
        for action in problem.actions(s):
            s_prime = problem.result(s,action)
            yield Node(state=s_prime,parent=self,action=action,g=problem.action_cost(self.g,s,action,s_prime))  
    
    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]
    
    def path(self):
        """Solution helper function. Returns the list of nodes forming 
        the path from the root to this node in order."""
        node = self
        sequence = []
        while node:
            sequence.append(node)
            node = node.parent
        return list(reversed(sequence))
    
    #Make node quickly hashable (state representation must be hashable of course)
    def __hash__(self):
        return hash(self.state)


# ____SEARCH ALGORITHMS___________________________________________________________________

#Implement best-first-search, which sees use in uniform cost, A*, and greedy searches
#The # of nodes generated = all reached nodes plus the duplicate nodes we tossed
def best_first_search(problem,f):
    root = Node(problem.initial)
    frontier = PriorityQueue(f)
    frontier.push(root)
    reached = {root.state:root}
    discarded = 0
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return discarded+len(reached),len(reached)-len(frontier)-1,node
        for child in node.expand(problem):
            if child.state not in reached:
                reached[child.state] = child
                frontier.push(child)
            elif f(child) < f(reached[child.state]):
                reached[child.state] = child
                del frontier[child]
                frontier.push(child)
                discarded += 1
    return discarded+len(reached),len(reached)-len(frontier)-1,None

#Iterative-deepening-DFS
# (pg 94 textbook)
def iterative_deepening_search(problem):
    depth = 0
    while True:
        if depth == 0:
            result = depth_limited_search(problem,depth,1)
        else:
            result = depth_limited_search(problem,depth,result[0])
        if result[2] != "cutoff":
                return result
        depth += 1

def depth_limited_search(problem,limit,ngen):
    root = Node(problem.initial)
    frontier = [root] 
    path = set()
    result = None
    while frontier:
        node = frontier.pop(0)
        if problem.goal_test(node.state):
            return ngen,len(path),node
        if node.depth > limit:
            result = "cutoff"
            path.discard(node.state)
        elif node.state not in path:
            for child in node.expand(problem):
                ngen += 1
                frontier.insert(0,child)
            path.add(node.state)
        else:
            path.discard(node.state)
    return ngen,len(path),result

"""  
def depth_limited_search(problem,limit,ngen):
    root = Node(problem.initial)
    frontier = [root]
    path = set()
    result = None #Fails if no solution at any depth
    while frontier:
        node = frontier.pop(0)
        if problem.goal_test(node.state):
            return ngen,len(path),node
        if node.depth > limit:
            result = "cutoff"
        elif node.state not in path:
            for child in node.expand(problem):
                frontier.insert(0,child)
                ngen += 1
            path.add(node.state)
        else:
            path.discard(node.state)

    return ngen,len(path),result
"""


#A* search
#Ties in f-cost could/should be broken by lowest h/highest g
def a_star_search(problem):
    """A* uses estimated final cost heuristic f=g+h; can call
    as a special case of best-first-search"""
    return best_first_search(problem,lambda n: problem.h(n)+n.g)

#Iterative-deepening-A*
def ida_star_search(problem):
    f_cost = problem.h(Node(problem.initial))
    ngen,nexp,result,f_cost = limited_astar_search(problem,f_cost,1)
    while True:
        ngen,nexp,result,f_cost = limited_astar_search(problem,f_cost,ngen)
        if result != "cutoff":
            return ngen,nexp,result

def limited_astar_search(problem,cutoff,ngen):
    root = Node(problem.initial)
    frontier = [root]
    path = set()
    next_cutoff = None
    result = None
    while frontier:
        node = frontier.pop(0)
        if problem.goal_test(node.state):
            return ngen,len(path),node,cutoff
        f_val = (problem.h(node)+node.g)
        if f_val > cutoff:
            if next_cutoff is None or f_val < next_cutoff:
                next_cutoff = f_val
            result = "cutoff"
            path.discard(node.state)
        elif node.state not in path:
            for child in node.expand(problem):
                frontier.insert(0,child)
                ngen += 1
            path.add(node.state)
        else:
            path.discard(node.state)
    return ngen,len(path),result,next_cutoff

# Greedy search
def greedy_search(problem):
    """A special case of best-first-search that first expands the node
        with the lowest h(n)"""
    return best_first_search(problem,problem.h)


#Old algorithms from A1 - DFS and UCS

def depth_first_search(problem):
    ngenerated = 1
    nexpanded = 0
    root = Node(problem.initial)
    frontier = [root] #Stack of nodes
    path = set() #lookup table used for cycle-checking along the current path from the root state:node
    
    while frontier:
        unique = 0 #tells us whether we're backtracking in the tree
        node = frontier.pop(0)
        path.add(node.state)
        if problem.goal_test(node.state):
            return ngenerated,nexpanded,node
        nexpanded += 1
        for child in node.expand(problem):
            ngenerated += 1
            if child.state not in path:
                frontier.insert(0,child)
                path.add(child.state)
                unique += 1
        if unique == 0: #no successor nodes => backtracking, so delete node from path
            path.discard(node.state)
    return ngenerated,nexpanded,None

def uniform_cost_search(problem):
    return best_first_search(problem,lambda n: n.g)

class PriorityQueue:
    def __init__(self,eval_fn):
        self.heap = []
        self.nodes = set()
        self.f = eval_fn
    
    def push(self,node):
        """Add a node to the PQeue, or replace a higher-cost version"""
        heapq.heappush(self.heap,[self.f(node),node])
        self.nodes.add(node)

    def pop(self):
        """Remove and return the minimum node in the PQueue"""
        self.nodes.discard(self.heap[0][1])
        return heapq.heappop(self.heap)[1] 

    def __getitem__(self,node):
        """Override the list index operator to only search for the node, 
        not the tuple of (cost,node). Returns the cost"""
        return next((v[0] for v in enumerate(self.heap) if v[1] == node), None)
        

    def __delitem__(self,node):
        """Enable item removal using just the node vs. tuple(cost,node).
        Used in replacing a node with a new lower-cost one"""
        if node not in self.nodes:
            return
        def getIndex():
            for pos,t in enumerate(self.heap):
                if t[1] == node:
                    return pos
        idx = getIndex()
        self.heap.pop(idx)
        self.nodes.discard(node)
        heapq.heapify(self.heap)
        

    def __contains__(self,node):
        """For use when checking if a node is in the frontier. This overrides
        the in/containment check operator so that the comparison is done on
        just the node, not tuple(cost,node)"""
        return node in self.nodes

    def __len__(self):
        return len(self.heap)
