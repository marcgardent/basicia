import logging
import random
from basicia.DeepEvolve.models import *
from basicia.DeepEvolve import Producter
from basicia.websocketenv import WebsocketEnv
from gym.spaces import Box
import basicia.ddpgagent as agent
import numpy as np

OBSERVATION = 2
ACTION = 2

class RunnerEnv(WebsocketEnv):
    action_space = Box(low=-1.0, high=1.0, shape=(ACTION,), dtype=np.float32)
    observation_space = Box(low=-100.0, high=100.0, shape=(OBSERVATION,), dtype=np.float32)
    reward_range = (0, 1)

def main_ga():
    
    hyperparams = agent.create_hyperparams_generator()
    env = RunnerEnv(None)
    callback = agent.TrainAndScoreCallback(env, agent.create_agent, dryrun=True)
    p = Producter(callback.train_and_score, hyperparams, 30)
    p.tournament(2)

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.DEBUG
    )
    main_ga()