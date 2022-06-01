import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # 经验回放的容量
        self.buffer = [] # 缓冲区
        self.position = 0 
    
    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # 随机采出小批量转移
        state, action, reward, next_state, done =  zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state, done
    
    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)
class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, device, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.device     = device
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        self.register_buffer('weight_epsilon', torch.FloatTensor(output_dim, input_dim))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(output_dim))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(output_dim))
        self.register_buffer('bias_epsilon', torch.FloatTensor(output_dim))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.device:
            weight_epsilon = self.weight_epsilon.cuda()
            bias_epsilon   = self.bias_epsilon.cuda()
        else:
            weight_epsilon = self.weight_epsilon
            bias_epsilon   = self.bias_epsilon
            
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.output_dim))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

class RainbowModel(nn.Module):
    def __init__(self, n_states, n_actions, n_atoms, Vmin, Vmax):
        super(RainbowModel, self).__init__()
        
        self.n_states   = n_states
        self.n_actions  = n_actions
        self.n_atoms    = n_atoms
        self.Vmin         = Vmin
        self.Vmax         = Vmax
        
        self.linear1 = nn.Linear(n_states, 32)
        self.linear2 = nn.Linear(32, 64)
        
        self.noisy_value1 = NoisyLinear(64, 64, device=device)
        self.noisy_value2 = NoisyLinear(64, self.n_atoms, device=device)
        
        self.noisy_advantage1 = NoisyLinear(64, 64, device=device)
        self.noisy_advantage2 = NoisyLinear(64, self.n_atoms * self.n_actions, device=device)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)
        
        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)
        
        value     = value.view(batch_size, 1, self.n_atoms)
        advantage = advantage.view(batch_size, self.n_actions, self.n_atoms)
        
        x = value + advantage - advantage.mean(1, keepdim=True)
        x = F.softmax(x.view(-1, self.n_atoms)).view(-1, self.n_actions, self.n_atoms)
        
        return x
        
    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()

    def act(self, state):
        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.n_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action

class RainbowDQN(nn.Module):
    def __init__(self, n_states, n_actions, n_atoms, Vmin, Vmax,cfg):
        super(RainbowDQN, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_atoms = cfg.n_atoms
        self.Vmin = cfg.Vmin
        self.Vmax = cfg.Vmax
        self.policy_model = RainbowModel(n_states, n_actions, n_atoms, Vmin, Vmax)
        self.target_model = RainbowModel(n_states, n_actions, n_atoms, Vmin, Vmax)
        self.batch_size = cfg.batch_size
        self.memory = ReplayBuffer(cfg.memory_capacity) # 经验回放
        self.optimizer = optim.Adam(self.policy_model.parameters(), 0.001)
    def choose_action(self,state):
        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        dist = self.policy_model(state).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.n_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action
    def projection_distribution(self,next_state, rewards, dones):
     
        
        delta_z = float(self.Vmax - self.Vmin) / (self.n_atoms - 1)
        support = torch.linspace(self.Vmin, self.Vmax, self.n_atoms)
        
        next_dist   = self.target_model(next_state).data.cpu() * support
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
        next_dist   = next_dist.gather(1, next_action).squeeze(1)
            
        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones   = dones.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)
        
        Tz = rewards + (1 - dones) * 0.99 * support
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
        b  = (Tz - self.Vmin) / delta_z
        l  = b.floor().long()
        u  = b.ceil().long()
            
        offset = torch.linspace(0, (self.batch_size - 1) * self.n_atoms, self.batch_size).long()\
                        .unsqueeze(1).expand(self.batch_size, self.n_atoms)

        proj_dist = torch.zeros(next_dist.size())    
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
            
        return proj_dist
    def update(self):
        if len(self.memory) < self.batch_size: # 当memory中不满足一个批量时，不更新策略
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size) 

        state      = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action     = Variable(torch.LongTensor(action))
        reward     = torch.FloatTensor(reward)
        done       = torch.FloatTensor(np.float32(done))

        proj_dist = self.projection_distribution(next_state, reward, done)
        
        dist = self.policy_model(state)
        action = action.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.n_atoms)
        dist = dist.gather(1, action).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        loss = -(Variable(proj_dist) * dist.log()).sum(1)
        loss  = loss.mean()
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.policy_model.reset_noise()
        self.target_model.reset_noise()
    