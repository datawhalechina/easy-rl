# Please do not make changes to this file - it will be overwritten with a clean
# version when your work is marked.
#
# This file contains code for the racetrack environment that you will be using
# as part of the second part of the CM50270: Reinforcement Learning coursework.

import time
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from IPython.display import clear_output

from matplotlib import colors

class RacetrackEnv(object) :
    """
    Class representing a race-track environment inspired by exercise 5.12 in Sutton & Barto 2018 (p.111).
    Please do not make changes to this class - it will be overwritten with a clean version when it comes to marking.

    The dynamics of this environment are detailed in this coursework exercise's jupyter notebook, although I have
    included rather verbose comments here  for those of you who are interested in how the environment has been
    implemented (though this should not impact your solution code).

    If you find any *bugs* with this code, please let me know immediately - thank you for finding them, sorry that I didn't!
    However, please do not suggest optimisations - some things have been purposely simplified for readability's sake.
    """


    ACTIONS_DICT = {
        0 : (1, -1),  # Acc Vert., Brake Horiz.
        1 : (1, 0),   # Acc Vert., Hold Horiz.
        2 : (1, 1),   # Acc Vert., Acc Horiz.
        3 : (0, -1),  # Hold Vert., Brake Horiz.
        4 : (0, 0),   # Hold Vert., Hold Horiz.
        5 : (0, 1),   # Hold Vert., Acc Horiz.
        6 : (-1, -1), # Brake Vert., Brake Horiz.
        7 : (-1, 0),  # Brake Vert., Hold Horiz.
        8 : (-1, 1)   # Brake Vert., Acc Horiz.
    }


    CELL_TYPES_DICT = {
        0 : "track",
        1 : "wall",
        2 : "start",
        3 : "goal"
    }


    def __init__(self) :
        # Load racetrack map from file.
        self.track = np.flip(np.loadtxt(os.path.dirname(__file__)+"/track.txt", dtype = int), axis = 0)


        # Discover start grid squares.
        self.initial_states = []
        for y in range(self.track.shape[0]) :
            for x in range(self.track.shape[1]) :
                if (self.CELL_TYPES_DICT[self.track[y, x]] == "start") :
                    self.initial_states.append((y, x))


        self.is_reset = False

        #print("Racetrack Environment File Loaded Successfully.")
        #print("Be sure to call .reset() before starting to initialise the environment and get an initial state!")


    def step(self, action : int) :
        """
        Takes a given action in the environment's current state, and returns a next state,
        reward, and whether the next state is terminal or not.

        Arguments:
            action {int} -- The action to take in the environment's current state. Should be an integer in the range [0-8].

        Raises:
            RuntimeError: Raised when the environment needs resetting.\n
            TypeError: Raised when an action of an invalid type is given.\n
            ValueError: Raised when an action outside the range [0-8] is given.\n

        Returns:
            A tuple of:\n
                {(int, int, int, int)} -- The next state, a tuple of (y_pos, x_pos, y_velocity, x_velocity).\n
                {int} -- The reward earned by taking the given action in the current environment state.\n
                {bool} -- Whether the environment's next state is terminal or not.\n

        """

        # Check whether a reset is needed.
        if (not self.is_reset) :
            raise RuntimeError(".step() has been called when .reset() is needed.\n" +
                               "You need to call .reset() before using .step() for the first time, and after an episode ends.\n" +
                               ".reset() initialises the environment at the start of an episode, then returns an initial state.")

        # Check that action is the correct type (either a python integer or a numpy integer).
        if (not (isinstance(action, int) or isinstance(action, np.integer))) :
            raise TypeError("action should be an integer.\n" +
                            "action value {} of type {} was supplied.".format(action, type(action)))

        # Check that action is an allowed value.
        if (action < 0 or action > 8) :
            raise ValueError("action must be an integer in the range [0-8] corresponding to one of the legal actions.\n" +
                             "action value {} was supplied.".format(action))


        # Update Velocity.
        # With probability, 0.85 update velocity components as intended.
        if (np.random.uniform() < 0.8) :
            (d_y, d_x) = self.ACTIONS_DICT[action]
        # With probability, 0.15 Do not change velocity components.
        else :
            (d_y, d_x) = (0, 0)

        self.velocity = (self.velocity[0] + d_y, self.velocity[1] + d_x)

		# Keep velocity within bounds (-10, 10).
        if (self.velocity[0] > 10) :
            self.velocity[0] = 10
        elif (self.velocity[0] < -10) :
            self.velocity[0] = -10
        if (self.velocity[1] > 10) :
            self.velocity[1] = 10
        elif (self.velocity[1] < -10) :
            self.velocity[1] = -10

        # Update Position.
        new_position = (self.position[0] + self.velocity[0], self.position[1] + self.velocity[1])

        reward = 0
        terminal = False

        # If position is out-of-bounds, return to start and set velocity components to zero.
        if (new_position[0] < 0 or new_position[1] < 0 or new_position[0] >= self.track.shape[0] or new_position[1] >= self.track.shape[1]) :
            self.position = random.choice(self.initial_states)
            self.velocity = (0, 0)
            reward -= 10
        # If position is in a wall grid-square, return to start and set velocity components to zero.
        elif (self.CELL_TYPES_DICT[self.track[new_position]] == "wall") :
            self.position = random.choice(self.initial_states)
            self.velocity = (0, 0)
            reward -= 10
        # If position is in a track grid-squre or a start-square, update position.
        elif (self.CELL_TYPES_DICT[self.track[new_position]] in ["track", "start"]) :
            self.position = new_position
        # If position is in a goal grid-square, end episode.
        elif (self.CELL_TYPES_DICT[self.track[new_position]] == "goal") :
            self.position = new_position
            reward += 10
            terminal = True
        # If this gets reached, then the student has touched something they shouldn't have. Naughty!
        else :
            raise RuntimeError("You've met with a terrible fate, haven't you?\nDon't modify things you shouldn't!")

        # Penalise every timestep.
        reward -= 1

        # Require a reset if the current state is terminal.
        if (terminal) :
            self.is_reset = False

        # Return next state, reward, and whether the episode has ended.
        return (self.position[0], self.position[1], self.velocity[0], self.velocity[1]), reward, terminal


    def reset(self) :
        """
        Resets the environment, ready for a new episode to begin, then returns an initial state.
        The initial state will be a starting grid square randomly chosen using a uniform distribution,
        with both components of the velocity being zero.

        Returns:
            {(int, int, int, int)} -- an initial state, a tuple of (y_pos, x_pos, y_velocity, x_velocity).
        """

        # Pick random starting grid-square.
        self.position = random.choice(self.initial_states)

        # Set both velocity components to zero.
        self.velocity = (0, 0)

        self.is_reset = True

        return (self.position[0], self.position[1], self.velocity[0], self.velocity[1])


    def render(self, sleep_time : float = 0.1) :
        """
        Renders a pretty matplotlib plot representing the current state of the environment.
        Calling this method on subsequent timesteps will update the plot.
        This is VERY VERY SLOW and wil slow down training a lot. Only use for debugging/testing.

        Arguments:
            sleep_time {float} -- How many seconds (or partial seconds) you want to wait on this rendered frame.

        """
        # Turn interactive mode on.
        plt.ion()
        fig = plt.figure(num = "env_render")
        ax = plt.gca()
        ax.clear()
        clear_output(wait = True)

        # Prepare the environment plot and mark the car's position.
        env_plot = np.copy(self.track)
        env_plot[self.position] = 4
        env_plot = np.flip(env_plot, axis = 0)

        # Plot the gridworld.
        cmap = colors.ListedColormap(["white", "black", "green", "red", "yellow"])
        bounds = list(range(6))
        norm = colors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(env_plot, cmap = cmap, norm = norm, zorder = 0)

        # Plot the velocity.
        if (not self.velocity == (0, 0)) :
            ax.arrow(self.position[1], self.track.shape[0] - 1 - self.position[0], self.velocity[1], -self.velocity[0],
                     path_effects=[pe.Stroke(linewidth=1, foreground='black')], color = "yellow", width = 0.1, length_includes_head = True, zorder = 2)

        # Set up axes.
        ax.grid(which = 'major', axis = 'both', linestyle = '-', color = 'k', linewidth = 2, zorder = 1)
        ax.set_xticks(np.arange(-0.5, self.track.shape[1] , 1));
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(-0.5, self.track.shape[0], 1));
        ax.set_yticklabels([])

        # Draw everything.
        #fig.canvas.draw()
        #fig.canvas.flush_events()

        plt.show()

        # Sleep if desired.
        if (sleep_time > 0) :
            time.sleep(sleep_time)


    def get_actions(self) :
        """
        Returns the available actions in the current state - will always be a list
        of integers in the range [0-8].
        """
        return [*self.ACTIONS_DICT]

# num_steps = 1000000

# env = RacetrackEnv()
# state = env.reset()
# print(state)

# for _ in range(num_steps) :

#     next_state, reward, terminal = env.step(random.choice(env.get_actions()))
#     print(next_state)
#     env.render()

#     if (terminal) :
#         _ = env.reset()
