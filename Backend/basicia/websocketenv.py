import websockets
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Env
from gym.spaces import Box
import asyncio
import nest_asyncio
import json

# fix - asyncio.run() Call async in sync evenif loop exists!
nest_asyncio.apply()

# fix crash CUBLAS  - https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)



class WebsocketEnv(Env):

    metadata = {'render.modes': []}
    action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) # # output shape, impulse (X,Y)
    observation_space = Box(low=0, high=1.0, shape=(8,), dtype=np.float32) # input shape : Vector3, (Velocity,Y+, Z+, RC(Y+),RayCast(..), ... )
    reward_range = (0, 1)
    spec = None
    done = False;

    def __init__(self, socket):
        self.websocket = socket

    def __send(self, obj):      
        print ("send", obj);
        payload= json.dumps(obj)
        asyncio.run(self.websocket.send(payload))
        
    
    def __receive(self):
        payload= asyncio.run(self.websocket.recv())
        return json.loads(payload);
        
    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        if self.done : return None
        # assert self.action_space.contains(action), "%r (%s) invalid" %(action, type(action))

        self.__send(action.tolist())
        
        return self.readState()

    def readState(self):
           
        state = self.__receive()
        self.done = state['done']
        # state['info'] <-- https://github.com/keras-rl/keras-rl/issues/264
        return (state['observation'], state['reward'], state['done'], dict({"fixme":0}))
        

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        self.done  = False
        asyncio.run(self.websocket.send("RESET"))
        return self.readState()[0];

    def render(self, mode='human'):

        pass

    def seed(self, seed=None):
        raise NotImplementedError

