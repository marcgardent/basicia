import tensorflow as tf
import tensorflowjs as tfjs
tf.compat.v1.disable_eager_execution()

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
from basicia.websocketenv import WebsocketEnv
import basicia.websocketserver 
import os.path
from keras.models import load_model

LAZYLOAD = True
TRAIN = False
STEPS = 50000
OBSERVATION = 2
ACTION = 2

class RunnerEnv(WebsocketEnv):
    action_space = Box(low=-1.0, high=1.0, shape=(ACTION,), dtype=np.float32)
    observation_space = Box(low=-100.0, high=100.0, shape=(OBSERVATION,), dtype=np.float32)
    reward_range = (0, 1)


def create_agent(env):
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    return actor, agent

def main(socket):
    
    # Get the environment and extract the number of actions.
    env = RunnerEnv(socket)
    
    # Next, we build a very simple model.
    
    actor, agent = create_agent(env)

    
    fname = "runner_goto_weights.h5f"
    filename, extension = os.path.splitext(fname)
    actor_filepath = filename + '_actor' + extension
    critic_filepath = filename + '_critic' + extension
    # load if exists
    if LAZYLOAD and os.path.isfile(actor_filepath) and os.path.isfile(critic_filepath) :
        print("loading....")
        agent.load_weights(fname)
        

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    if TRAIN:
        agent.fit(env, nb_steps=STEPS, visualize=False, verbose=1, nb_max_episode_steps=1000)
        agent.save_weights(fname,overwrite=True)
    
    # export as tensorflow.js https://www.tensorflow.org/js/tutorials/conversion/import_keras?hl=fr
    tfjs.converters.save_keras_model(actor, "runner-goto-tfjs")

    # Finally, evaluate our algorithm for 50 episodes.
    agent.test(env, nb_episodes=50, visualize=True, nb_max_episode_steps=1000)


if __name__ == "__main__":
    # execute only if run as a script
    basicia.websocketserver.start(main)