
class DefaultConfig:
    def __init__(self) -> None:
        pass
    def print_cfg(self):
        print(self.__dict__)
class GeneralConfig(DefaultConfig):
    def __init__(self) -> None:
        self.env_name = "CartPole-v1" # name of environment
        self.algo_name = "DQN" # name of algorithm
        self.mode = "train" # train or test
        self.seed = 0 # random seed
        self.device = "cuda" # device to use
        self.train_eps = 200 # number of episodes for training
        self.test_eps = 20 # number of episodes for testing
        self.eval_eps = 10 # number of episodes for evaluation
        self.eval_per_episode = 5 # evaluation per episode
        self.max_steps = 200 # max steps for each episode
        self.load_checkpoint = False
        self.load_path = None # path to load model
        self.show_fig = False # show figure or not
        self.save_fig = True # save figure or not
        
class AlgoConfig(DefaultConfig):
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        # self.epsilon_start = 0.95 # epsilon start value
        # self.epsilon_end = 0.01 # epsilon end value
        # self.epsilon_decay = 500 # epsilon decay rate
        self.gamma = 0.95 # discount factor
        # self.lr = 0.0001 # learning rate
        # self.buffer_size = 100000 # size of replay buffer
        # self.batch_size = 64 # batch size
        # self.target_update = 4 # target network update frequency
class MergedConfig:
    def __init__(self) -> None:
        pass
        