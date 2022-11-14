from common.utils import get_logger,save_results,save_cfgs,plot_rewards,merge_class_attrs,load_cfgs
from common.config import GeneralConfig,AlgoConfig,MergedConfig
import time
from pathlib import Path
import datetime
import argparse

class Launcher:
    def __init__(self) -> None:
        self.get_cfg()
    def get_cfg(self):
        self.cfgs = {'general_cfg':GeneralConfig(),'algo_cfg':AlgoConfig()}  # create config 
    def process_yaml_cfg(self):
        ''' load yaml config
        '''
        parser = argparse.ArgumentParser(description="hyperparameters")  
        parser.add_argument('--yaml', default = None, type=str,help='the path of config file')
        args = parser.parse_args()
        if args.yaml is not None:
            load_cfgs(self.cfgs, args.yaml)
    def print_cfg(self,cfg):
        ''' print parameters
        '''
        cfg_dict = vars(cfg)
        print("Hyperparameters:")
        print(''.join(['=']*80))
        tplt = "{:^20}\t{:^20}\t{:^20}"
        print(tplt.format("Name", "Value", "Type"))
        for k,v in cfg_dict.items():
            print(tplt.format(k,v,str(type(v))))   
        print(''.join(['=']*80))
    def env_agent_config(self,cfg,logger):
        env,agent = None,None
        return env,agent
    def train_one_episode(self,env, agent, cfg):
        ep_reward = 0
        ep_step = 0
        return agent,ep_reward,ep_step
    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0
        ep_step = 0
        return agent,ep_reward,ep_step
    def evaluate(self, env, agent, cfg):
        sum_eval_reward = 0
        for _ in range(cfg.eval_eps):
            _,eval_ep_reward,_ = self.test_one_episode(env, agent, cfg)
            sum_eval_reward += eval_ep_reward
        mean_eval_reward = sum_eval_reward/cfg.eval_eps
        return mean_eval_reward
    # def train(self,cfg, env, agent,logger):
    #     res_dic = {}
    #     return res_dic
    # def test(self,cfg, env, agent,logger):
    #     res_dic = {}
    #     return res_dic
    def create_path(self,cfg):
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")   # obtain current time
        self.task_dir = f"{cfg.mode.capitalize()}_{cfg.env_name}_{cfg.algo_name}_{curr_time}"
        Path(self.task_dir).mkdir(parents=True, exist_ok=True)
        self.model_dir = f"{self.task_dir}/models/"
        self.res_dir = f"{self.task_dir}/results/"
        self.log_dir = f"{self.task_dir}/logs/"
    def run(self):
        self.process_yaml_cfg() # load yaml config
        cfg = MergedConfig() # merge config
        cfg = merge_class_attrs(cfg,self.cfgs['general_cfg'])
        cfg = merge_class_attrs(cfg,self.cfgs['algo_cfg'])
        self.print_cfg(cfg) # print the configuration
        self.create_path(cfg) # create the path to save the results
        logger = get_logger(self.log_dir) # create the logger
        env, agent = self.env_agent_config(cfg,logger)
        if cfg.load_checkpoint:
            agent.load_model(f"{cfg.load_path}/models/")
        logger.info(f"Start {cfg.mode}ing!")
        logger.info(f"Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
        rewards = []  # record rewards for all episodes
        steps = [] # record steps for all episodes
        if cfg.mode.lower() == 'train':
            best_ep_reward = -float('inf')
            for i_ep in range(cfg.train_eps):
                agent,ep_reward,ep_step = self.train_one_episode(env, agent, cfg)
                logger.info(f"Episode: {i_ep+1}/{cfg.train_eps}, Reward: {ep_reward:.3f}, Step: {ep_step}")
                rewards.append(ep_reward)
                steps.append(ep_step)
                # for _ in range
                if (i_ep+1)%cfg.eval_per_episode == 0:
                    mean_eval_reward = self.evaluate(env, agent, cfg)
                    if mean_eval_reward  >= best_ep_reward: # update best reward
                        logger.info(f"Current episode {i_ep+1} has the best eval reward: {mean_eval_reward:.3f}")
                        best_ep_reward = mean_eval_reward 
                        agent.save_model(self.model_dir) # save models with best reward
            # env.close()
        elif cfg.mode.lower() == 'test':
            for i_ep in range(cfg.test_eps):
                agent,ep_reward,ep_step = self.test_one_episode(env, agent, cfg)
                logger.info(f"Episode: {i_ep+1}/{cfg.test_eps}, Reward: {ep_reward:.3f}, Step: {ep_step}")
                rewards.append(ep_reward)
                steps.append(ep_step)
            agent.save_model(self.model_dir)  # save models
            # env.close()
        logger.info(f"Finish {cfg.mode}ing!")
        res_dic = {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}
        save_results(res_dic, self.res_dir) # save results
        save_cfgs(self.cfgs, self.task_dir) # save config
        plot_rewards(rewards, title=f"{cfg.mode.lower()}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}" ,fpath= self.res_dir)
    # def run(self):
    #     self.process_yaml_cfg() # load yaml config
    #     cfg = MergedConfig() # merge config
    #     cfg = merge_class_attrs(cfg,self.cfgs['general_cfg'])
    #     cfg = merge_class_attrs(cfg,self.cfgs['algo_cfg'])
    #     self.print_cfg(cfg) # print the configuration
    #     self.create_path(cfg) # create the path to save the results
    #     logger = get_logger(self.log_dir) # create the logger
    #     env, agent = self.env_agent_config(cfg,logger)
    #     if cfg.load_checkpoint:
    #         agent.load_model(f"{cfg.load_path}/models/")
    #     if cfg.mode.lower() == 'train':
    #         res_dic = self.train(cfg, env, agent,logger)
    #     elif cfg.mode.lower() == 'test':
    #         res_dic = self.test(cfg, env, agent,logger)
    #     save_results(res_dic, self.res_dir) # save results
    #     save_cfgs(self.cfgs, self.task_dir) # save config
    #     agent.save_model(self.model_dir)  # save models
    #     plot_rewards(res_dic['rewards'], title=f"{cfg.mode.lower()}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}" ,fpath= self.res_dir)