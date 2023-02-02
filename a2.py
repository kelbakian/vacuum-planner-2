import sys
import argparse
import search
import collections
from dataclasses import dataclass, field

#World's symbolic representation
wall = '#'
dirt = '*'
charger = ':'
start = '@'

@dataclass
class Environment():
    """Represents the static problem/world info.
    Holds world dims, action space, obstacle list, and fuel capacity"""
    dimrow: int = 0
    dimcol: int = 0
    possible_actions: "set[str]" = field(default_factory=set)
    obstacles: "set[tuple()]" = field(default_factory=set)
    chargers: "set[tuple()]" = field(default_factory=set)
    fuel_capacity: int = None



@dataclass()
class StateRepresentation():
    #Vacuum robot planner state representation
    loc: tuple() = field(default_factory=tuple)
    dirt_list: "set[tuple()]" = field(default_factory=set)
    remaining_fuel: int = None

    #Compare states using dirt distribution.
    def __eq__(self, other):
        return (
            getattr(other, 'loc', None) == self.loc and
            collections.Counter(getattr(other,'dirt_list',None)) == collections.Counter(self.dirt_list) and
            getattr(other,'remaining_fuel',None)==self.remaining_fuel)
    
    def __ne__(self,other):
        return not self == other

    #Define the hash value for this datatype so that it can be used in dicts,sets,etc.
    def __hash__(self):
        if self.remaining_fuel is not None:
            return hash((self.loc,tuple(self.dirt_list),self.remaining_fuel))
        else:
            return hash((self.loc,tuple(self.dirt_list)))
    


def main(*args):
    #Parse user commands
    parser = argparse.ArgumentParser()
    parser.add_argument("algo")
    parser.add_argument("-battery","--battery",action="store_true",required=False)
    parser.add_argument("heuristic",nargs='?',choices=['h0','h1','h2','h3'],default=None)
    args = parser.parse_args()
    
    algo_choices = {"uniform-cost","depth-first","depth-first-id","a-star","ida-star","greedy"}   
    if args.algo not in algo_choices:
        sys.exit("Invalid search algo.")
    
    if (args.algo == "a-star" or args.algo=="ida-star" or args.algo=="greedy"):
        if args.heuristic == None:
            sys.exit("Must choose a heuristic for A* or greedy search")

    #World dimensions - deterministically given this way
    ncol = int(sys.stdin.readline())
    nrow = int(sys.stdin.readline())

    obstacles = set()
    dirt_list =set()
    chargers = set()

    for row in range(nrow):
        for col in range(ncol):
            cell = sys.stdin.read(1)
            if cell == '\n':
                cell = sys.stdin.read(1)
            if cell == start:
                initial_loc = (row,col)
            elif cell == dirt:
                dirt_list.add((row,col))
            elif cell == charger:
                chargers.add((row,col))
            elif cell == wall:
                obstacles.add((row,col))
    

    #Formulate and solve the problem:
    if args.battery:
        actions = ['N','S','E','W','V','R']
        fuel = ((nrow-1)+(ncol-1))*2+1
        vacuum_world = Environment(nrow-1,ncol-1,actions,obstacles,chargers,fuel)
        initial_state = StateRepresentation(initial_loc,dirt_list,fuel)
    else:
        actions = ['N','S','E','W','V']
        vacuum_world = Environment(nrow-1,ncol-1,actions,obstacles,chargers)
        initial_state = StateRepresentation(initial_loc,dirt_list)

    
    #Call the agent and get solution, or none if DNE
    agent = search.SimpleVacuumAgent(vacuum_world,initial_state,args.algo,args.heuristic)
    ngen, nexp, path = agent()
    if path is not None:
        print(*path.solution(),sep="\n")
    else:
        print("No solution found")
    print(f"{ngen} nodes generated")
    print(f"{nexp} nodes expanded")


if __name__ == "__main__":
    main()