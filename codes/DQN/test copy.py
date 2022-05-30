import random
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import gym
import time
from collections import deque
from tensorflow.keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Activation, Flatten, Conv1D, MaxPooling1D,Reshape
import matplotlib.pyplot as plt

class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=400000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay =  self.epsilon_min / 500000
        
        self.batch_size = 32
        self.train_start = 1000
        self.state_size = self.env.observation_space.shape[0]*4
        self.action_size = self.env.action_space.n
        self.learning_rate = 0.00025
        
        self.evaluation_model = self.create_model()
        self.target_model = self.create_model()
        
    def create_model(self):
        model = Sequential()
        model.add(Dense(128*2, input_dim=self.state_size,activation='relu'))
        model.add(Dense(128*2, activation='relu'))
        model.add(Dense(128*2, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=self.learning_rate,decay=0.99,epsilon=1e-6))
        return model
    
    def choose_action(self, state, steps):
        if steps > 50000:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.evaluation_model.predict(state)[0])
        
    def remember(self, cur_state, action, reward, new_state, done):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        transition = (cur_state, action, reward, new_state, done)
        self.memory.extend([transition])
        
        self.memory_counter += 1
    
    def replay(self):
        if len(self.memory) < self.train_start:
            return
        
        mini_batch = random.sample(self.memory, self.batch_size)
        
        update_input = np.zeros((self.batch_size, self.state_size))
        update_target = np.zeros((self.batch_size, self.action_size))
        
        for i in range(self.batch_size):
            state, action, reward, new_state, done = mini_batch[i]
            target = self.evaluation_model.predict(state)[0]
        
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * np.amax(self.target_model.predict(new_state)[0])
            
            update_input[i] = state
            update_target[i] = target
    
        self.evaluation_model.fit(update_input, update_target, batch_size=self.batch_size, epochs=1, verbose=0)
    
    def target_train(self):
        self.target_model.set_weights(self.evaluation_model.get_weights())
        return
    
    def visualize(self, reward, episode):
        plt.plot(episode, reward, 'ob-')
        plt.title('Average reward each 100 episode')
        plt.ylabel('Reward')
        plt.xlabel('Episodes')
        plt.grid()
        plt.show()
    
    def transform(self,state):
        if state.shape[1]==512:
            return state
        a=[np.binary_repr(x,width=8) for x in state[0]]
        res=[]
        for x in a:
            res.extend([x[:2],x[2:4],x[4:6],x[6:]])
        res=[int(x,2) for x in res]
        return np.array(res)
        
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main():
    # env = gym.make('Breakout-ram-v0')
    env = gym.make('Breakout-ram-v0')
    env = env.unwrapped
    
    print(env.action_space)
    print(env.observation_space.shape[0])
    print(env.observation_space.high)
    print(env.observation_space.low)
    
    #print(env.observation_space.shape)
    
    
    episodes = 5000
    trial_len = 10000
    
    tmp_reward=0
    sum_rewards = 0
    n_success = 0
    total_steps = 0
    
    graph_reward = []
    graph_episodes = []
    time_record = []
    
    dqn_agent = DQN(env=env)
    for i_episode in range(episodes):
        start_time = time.time()
        total_reward = 0
        cur_state = env.reset().reshape(1,128)
        cur_state=dqn_agent.transform(cur_state).reshape(1,128*4)/4
        i_step=0
        for step in range(trial_len):
            #env.render()
            i_step+=1
            action = dqn_agent.choose_action(cur_state, total_steps)
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1, 128)
            new_state = dqn_agent.transform(new_state).reshape(1,128*4)/4
            total_reward += reward
            sum_rewards += reward
            tmp_reward += reward
            if reward>0:    #Testing whether it is good.
                reward=1
            
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            if total_steps > 10000:
                if total_steps%4 == 0:
                    dqn_agent.replay()
                if total_steps%5000 == 0:
                    dqn_agent.target_train()
            
            cur_state = new_state
            total_steps += 1
            if done:
                env.reset()
                break
        if (i_episode+1) % 100 == 0:
            graph_reward.append(sum_rewards/100)
            graph_episodes.append(i_episode+1)
            sum_rewards = 0
            print("Episode ",i_episode+1," Reward: ")
            print(graph_reward[-1])
        end_time = time.time()
        time_record.append(end_time-start_time)
        print("NOW in episode: " + str(i_episode))
        print("Time cost: " + str(end_time-start_time))
        print("Reward: ",tmp_reward)
        print("Step:", i_step)
        tmp_reward=0
    print("Reward: ")
    print(graph_reward)
    print("Episode: ")
    print(graph_episodes)
    print("Average_time: ")
    print(sum(time_record)/5000)
    dqn_agent.visualize(graph_reward, graph_episodes)
    
if __name__ == '__main__':
    main()