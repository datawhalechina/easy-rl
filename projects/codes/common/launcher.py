from common.utils import save_args,save_results,plot_rewards
class Launcher:
    def __init__(self) -> None:
        pass
    def get_args(self):
        cfg = {}
        return cfg
    def env_agent_config(self,cfg):
        env,agent = None,None
        return env,agent
    def train(self,cfg, env, agent):
        res_dic = {}
        return res_dic
    def test(self,cfg, env, agent):
        res_dic = {}
        return res_dic

    def run(self):
        cfg = self.get_args()
        env, agent = self.env_agent_config(cfg)
        res_dic = self.train(cfg, env, agent)
        save_args(cfg,path = cfg['result_path']) # save parameters
        agent.save_model(path = cfg['model_path'])  # save models
        save_results(res_dic, tag = 'train', path = cfg['result_path']) # save results
        plot_rewards(res_dic['rewards'], cfg, path = cfg['result_path'],tag = "train")  # plot results
        # testing
        # env, agent = self.env_agent_config(cfg) # create new env for testing, sometimes can ignore this step
        agent.load_model(path = cfg['model_path'])  # load model
        res_dic = self.test(cfg, env, agent)
        save_results(res_dic, tag='test',
                    path = cfg['result_path'])  
        plot_rewards(res_dic['rewards'], cfg, path = cfg['result_path'],tag = "test") 
