import numpy as np
import utils
import random
import math


class Agent:

    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        self.reset()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None
                
    def f(self,u,n):
        if n < self.Ne:
            return 1
        return u

    def R(self,points,dead):
        if dead:
            return -1
        elif points > self.points:
            return 1
        return -0.1

    def get_state(self, state):
        # [adjoining_wall_x, adjoining_wall_y]
        adjoining_wall_x = int(state[0] == utils.WALL_SIZE) + 2 * int(state[0] == utils.DISPLAY_SIZE - utils.WALL_SIZE)
        adjoining_wall_y = int(state[1] == utils.WALL_SIZE) + 2 * int(state[1] == utils.DISPLAY_SIZE - utils.WALL_SIZE)
        # [food_dir_x, food_dir_y] 
        food_dir_x = 1 + int(state[0] < state[3]) - int(state[0] == state[3])
        food_dir_y = 1 + int(state[1] < state[4]) - int(state[1] == state[4])
        # [adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right]
        adjoining_body = [(state[0] - body_state[0], state[1] - body_state[1]) for body_state in state[2]]
        adjoining_body_top = int([0, utils.GRID_SIZE] in adjoining_body)
        adjoining_body_bottom = int([0, -utils.GRID_SIZE] in adjoining_body)
        adjoining_body_left = int([utils.GRID_SIZE, 0] in adjoining_body)
        adjoining_body_right = int([-utils.GRID_SIZE, 0] in adjoining_body)
        return adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right


    def update(self, _state, points, dead):
        if self.s:
            maxq = max(self.Q[_state]) 
            reward = self.R(points,dead)
            alpha = self.C / (self.C + self.N[self.s][self.a])
            self.Q[self.s][self.a] += alpha * (reward + self.gamma * maxq - self.Q[self.s][self.a])
            self.N[self.s][self.a] += 1.0
        
    def choose_action(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)
        '''
        
        _state = self.get_state(state)
        Qs = self.Q[_state][:]
        
        if self._train:
            self.update(_state, points, dead)
            if dead:
                self.reset()  
                return
            Ns = self.N[_state]
            Fs = [self.f(Qs[a], Ns[a]) for a in self.actions]
            action = np.argmax(Fs)
            self.s = _state
            self.a = action   
        else:
            if dead:
                self.reset()  
                return
            action = np.argmax(Qs)

        self.points = points             
        return action
