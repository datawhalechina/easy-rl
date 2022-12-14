#!/usr/bin/env python

# simple_grid.py
# based on frozen_lake.py
# adapted by Frans Oliehoek.
# 
import sys
from contextlib import closing

import numpy as np
from io import StringIO
#from six import StringIO, b
import gym
from gym import utils
from gym import Env, spaces
from gym.utils import seeding


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class DiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)

    (*) dictionary of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS


    """

    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None  # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return int(self.s)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, {"prob": p})
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "theAlley": [
        "S...H...H...G"
    ],
    "walkInThePark": [
        "S.......",
        ".....H..",
        "........",
        "......H.",
        "........",
        "...H...G"
    ],
    "1Dtest": [

    ],
    "4x4": [
        "S...",
        ".H.H",
        "...H",
        "H..G"
    ],
    "8x8": [
        "S.......",
        "........",
        "...H....",
        ".....H..",
        "...H....",
        ".HH...H.",
        ".H..H.H.",
        "...H...G"
    ],
}

POTHOLE_PROB = 0.2
BROKEN_LEG_PENALTY = -5
SLEEP_DEPRIVATION_PENALTY = -0.0
REWARD = 10

def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0,0))
        while frontier:
            r, c = frontier.pop()
            if not (r,c) in discovered:
                discovered.add((r,c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] not in '#H'):
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['.', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


class DrunkenWalkEnv(DiscreteEnv):
    """
    A simple grid environment, completely based on the code of 'FrozenLake', credits to 
    the original authors.

    You're finding your way home (G) after a great party which was happening at (S).
    Unfortunately, due to recreational intoxication you find yourself only moving into 
    the intended direction 80% of the time, and perpendicular to that the other 20%.

    To make matters worse, the local community has been cutting the budgets for pavement
    maintenance, which means that the way to home is full of potholes, which are very likely
    to make you trip. If you fall, you are obviously magically transported back to the party, 
    without getting some of that hard-earned sleep.

        S...
        .H.H
        ...H
        H..G

    S : starting point
    . : normal pavement
    H : pothole, you have a POTHOLE_PROB chance of tripping
    G : goal, time for bed

    The episode ends when you reach the goal or trip.
    You receive a reward of +10 if you reach the goal, 
    but get a SLEEP_DEPRIVATION_PENALTY and otherwise.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4",is_slippery=True):
        """ This generates a map and sets all transition probabilities.

            (by passing constructed nS, nA, P, isd to DiscreteEnv)
        """
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]

        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        # We need to pass 'P' to DiscreteEnv:
        # P dictionary dict of dicts of lists, where
        # P[s][a] == [(probability, nextstate, reward, done), ...]
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def convert_rc_to_s(row, col):
            return row*ncol + col

        #def inc(row, col, a):
        def intended_destination(row, col, a):
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            return (row, col)

        def construct_transition_for_intended(row, col, a, prob, li):
            """ this constructs a transition to the "intended_destination(row, col, a)"
                and adds it to the transition list (which could be for a different action b).

            """
            newrow, newcol = intended_destination(row, col, a)
            newstate = convert_rc_to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            done = bytes(newletter) in b'G'
            rew = REWARD if newletter == b'G' else SLEEP_DEPRIVATION_PENALTY
            li.append( (prob, newstate, rew, done) )


        #THIS IS WHERE THE MATRIX OF TRANSITION PROBABILITIES IS COMPUTED.
        for row in range(nrow):
            for col in range(ncol):
                # specify transitions for s=(row, col)
                s = convert_rc_to_s(row, col)
                letter = desc[row, col]
                for a in range(4):
                    # specify transitions for action a
                    li = P[s][a]
                    if letter in b'G':
                        # We are at the goal ('G').... 
                        # This is a strange case:
                        # - conceptually, we can think of this as:
                        #     always transition to a 'terminated' state where we willget 0 reward.
                        #
                        # - But in gym, in practie, this case should not be happening at all!!!
                        #   Gym will alreay have returned 'done' when transitioning TO the goal state (not from it).
                        #   So we will never use the transition probabilities *from* the goal state.
                        #   So, from gym's perspective we could specify anything we like here. E.g.,:
                        #       li.append((1.0, 59, 42000000, True))
                        #
                        # However, if we want to be able to use the transition matrix to do value iteration, it is important
                        # that we get 0 reward ever after.
                        li.append((1.0, s, 0, True))

                    if letter in b'H':
                        #We are at a pothole ('H')
                        #when we are at a pothole, we trip with prob. POTHOLE_PROB
                        li.append((POTHOLE_PROB, s, BROKEN_LEG_PENALTY, True))
                        construct_transition_for_intended(row, col, a, 1.0 - POTHOLE_PROB, li)
                        
                    else:
                        # We are at normal pavement (.)
                        # with prob. 0.8 we move as intended:
                        construct_transition_for_intended(row, col, a, 0.8, li)
                        # but with prob. 0.1 we move sideways to intended:
                        for b in [(a-1)%4, (a+1)%4]:
                            construct_transition_for_intended(row, col, b, 0.1, li)

        super(DrunkenWalkEnv, self).__init__(nS, nA, P, isd)

    def action_to_string(self, action_index):
        s ="{}".format(["Left","Down","Right","Up"][action_index])
        return s

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(" (last action was '{action}')\n".format( action=self.action_to_string(self.lastaction) ) )
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
if __name__ == "__main__":
    # env = DrunkenWalkEnv(map_name="walkInThePark")
    env = DrunkenWalkEnv(map_name="theAlley")
    n_states = env.observation_space.n
    n_actions = env.action_space.n