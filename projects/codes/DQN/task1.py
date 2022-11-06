#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-10-24 08:21:31
LastEditor: JiangJi
LastEditTime: 2022-10-26 09:50:49
Discription: Not finished
'''
import sys,os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE" # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
sys.path.append(parent_path)  # add path to system path

import gym
import torch
import datetime
import numpy as np
import argparse
from common.utils import all_seed
from common.models import MLP
from common.memories import ReplayBuffer
from common.launcher import Launcher
from envs.register import register_env
from dqn import DQN
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

# xvfb-run -s "-screen 0 640x480x24" python main1.py 
def get_cart_location(env,screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render().transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env,screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen)


class CNN(nn.Module):

    def __init__(self, h, w, outputs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
class Main(Launcher):
    def get_args(self):
        """ hyperparameters
        """
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
        parser = argparse.ArgumentParser(description="hyperparameters")      
        parser.add_argument('--algo_name',default='DQN',type=str,help="name of algorithm")
        parser.add_argument('--env_name',default='CartPole-v1',type=str,help="name of environment")
        parser.add_argument('--train_eps',default=800,type=int,help="episodes of training")
        parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
        parser.add_argument('--ep_max_steps',default = 100000,type=int,help="steps per episode, much larger value can simulate infinite steps")
        parser.add_argument('--gamma',default=0.999,type=float,help="discounted factor")
        parser.add_argument('--epsilon_start',default=0.95,type=float,help="initial value of epsilon")
        parser.add_argument('--epsilon_end',default=0.01,type=float,help="final value of epsilon")
        parser.add_argument('--epsilon_decay',default=500,type=int,help="decay rate of epsilon, the higher value, the slower decay")
        parser.add_argument('--lr',default=0.0001,type=float,help="learning rate")
        parser.add_argument('--memory_capacity',default=100000,type=int,help="memory capacity")
        parser.add_argument('--batch_size',default=128,type=int)
        parser.add_argument('--target_update',default=4,type=int)
        parser.add_argument('--hidden_dim',default=256,type=int)
        parser.add_argument('--device',default='cuda',type=str,help="cpu or cuda") 
        parser.add_argument('--seed',default=10,type=int,help="seed") 
        parser.add_argument('--show_fig',default=False,type=bool,help="if show figure or not")  
        parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")
        # please manually change the following args in this script if you want
        parser.add_argument('--result_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                '/' + curr_time + '/results' )
        parser.add_argument('--model_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                '/' + curr_time + '/models' )    
        args = parser.parse_args()    
        args = {**vars(args)}  # type(dict)         
        return args

    def env_agent_config(self,cfg):
        ''' create env and agent
        '''
        env = gym.make('CartPole-v1', new_step_api=True, render_mode='single_rgb_array').unwrapped
        if cfg['seed'] !=0: # set random seed
            all_seed(env,seed=cfg["seed"]) 
        try: # state dimension
            n_states = env.observation_space.n # print(hasattr(env.observation_space, 'n'))
        except AttributeError:
            n_states = env.observation_space.shape[0] # print(hasattr(env.observation_space, 'shape'))
        n_actions = env.action_space.n  # action dimension
        print(f"n_states: {n_states}, n_actions: {n_actions}")
        cfg.update({"n_states":n_states,"n_actions":n_actions}) # update to cfg paramters
        env.reset()
        init_screen = get_screen(env)
        _, screen_height, screen_width = init_screen.shape
        model = CNN(screen_height, screen_width, n_actions)
        memory =  ReplayBuffer(cfg["memory_capacity"]) # replay buffer
        agent = DQN(model,memory,cfg)  # create agent
        return env, agent

    def train(self,cfg, env, agent):
        ''' 训练
        '''
        print("Start training!")
        print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
        rewards = []  # record rewards for all episodes
        steps = []
        for i_ep in range(cfg["train_eps"]):
            ep_reward = 0  # reward per episode
            ep_step = 0
            state = env.reset()  # reset and obtain initial state
            last_screen = get_screen(env)
            current_screen = get_screen(env)
            state = current_screen - last_screen
            for _ in range(cfg['ep_max_steps']):
                ep_step += 1
                action = agent.sample_action(state)  # sample action
                _, reward, done, _,_  = env.step(action)  # update env and return transitions
                last_screen = current_screen
                current_screen = get_screen(env)
                next_state = current_screen - last_screen
                agent.memory.push(state.cpu().numpy(), action, reward,
                                next_state.cpu().numpy(), done)  # save transitions
                state = next_state  # update next state for env
                agent.update()  # update agent
                ep_reward += reward  #
                if done:
                    break
            if (i_ep + 1) % cfg["target_update"] == 0:  # target net update, target_update means "C" in pseucodes
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            steps.append(ep_step)
            rewards.append(ep_reward)
            if (i_ep + 1) % 10 == 0:
                print(f'Episode: {i_ep+1}/{cfg["train_eps"]}, Reward: {ep_reward:.2f}, step: {ep_step:d}, Epislon: {agent.epsilon:.3f}')
        print("Finish training!")
        env.close()
        res_dic = {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}
        return res_dic

    def test(self,cfg, env, agent):
        print("Start testing!")
        print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
        rewards = []  # record rewards for all episodes
        steps = []
        for i_ep in range(cfg['test_eps']):
            ep_reward = 0  # reward per episode
            ep_step = 0
            state = env.reset()  # reset and obtain initial state
            last_screen = get_screen(env)
            current_screen = get_screen(env)
            state = current_screen - last_screen
            for _ in range(cfg['ep_max_steps']):
                ep_step+=1
                action = agent.predict_action(state)  # predict action
                _, reward, done, _,_ = env.step(action)  
                last_screen = current_screen
                current_screen = get_screen(env)
                next_state = current_screen - last_screen
                state = next_state  
                ep_reward += reward 
                if done:
                    break
            steps.append(ep_step)
            rewards.append(ep_reward)
            print(f"Episode: {i_ep+1}/{cfg['test_eps']}，Reward: {ep_reward:.2f}")
        print("Finish testing!")
        env.close()
        return {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}


if __name__ == "__main__":
    main = Main()
    main.run()
