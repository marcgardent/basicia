import logging
import random
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.models import load_model
import keras.backend as K
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Env
from gym.spaces import Box
from .DeepEvolve.models import *

def create_agent(observation_space, action_space, hyperparams):
    assert len(action_space.shape) == 1
    nb_actions = action_space.shape[0]

    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + observation_space.shape))
    for layer in hyperparams['actor']['layers']:
        actor.add(Dense(layer['neurons'].value))
        actor.add(Activation(layer['activation'].value))
    
    actor.add(Dense(nb_actions))
    actor.add(Activation(hyperparams['actor']['output_activation'].value))
    

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + observation_space.shape, name='observation_input')
    
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    for layer in hyperparams['critic']['layers']:
        x = Dense(layer['neurons'].value, activation=layer['activation'].value)(x)
    x = Dense(1, activation=hyperparams['critic']['output_activation'].value)(x)

    critic = Model(inputs=[action_input, observation_input], outputs=x)
    

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(optimizer=hyperparams['optimizer'].value , metrics=['mae'])

    return actor, agent


def create_hyperparams_generator():
    neurons = DiscretGene((8, 16, 32, 64, 128, 256, 512, 768, 1024))
    activation = DiscretGene(('relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'))
    optimizer = DiscretGene(('rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam'))
    layer = ComplexNode({
        "activation" : activation.clone(),
        "neurons" : neurons.clone()
        })

    nn = ComplexNode({
            "layers" : ArrayNode.Duplicate(layer, 1, 6),
            "output_activation" : activation.clone()
        })
    
    ddpgagent = ComplexNode({
        "actor" : nn.clone(),
        "critic" : nn.clone(),
        "optimizer" : optimizer.clone()
    })

    return ddpgagent

class TrainAndScoreCallback:

    def __init__(self, env, create_agent, dryrun=False):
        self.env = env
        self.create_agent = create_agent
        self.dryrun = dryrun

    def train_and_score(self, hyperparams):
        if self.dryrun:
            return random.random()
        else:
            actor, agent = self.create_agent(self.env.observation_space, self.env.action_space, hyperparams)
            self.train(agent)
            ret = self.score(agent)
            K.clear_session()
            return ret
    
    def train(self, agent):
        TOTAL_STEPS = 1000
        STEP_MAX_BY_EPISODE = 400
        agent.fit(self.env, nb_steps=TOTAL_STEPS , visualize=False, verbose=1, nb_max_episode_steps=STEP_MAX_BY_EPISODE )

    def score(self, agent):
        STEP_MAX_BY_EPISODE=200
        EPISODES = 20

        # Finally, evaluate our algorithm for 50 episodes.
        history = agent.test(self.env, nb_episodes=EPISODES, visualize=True, nb_max_episode_steps=STEP_MAX_BY_EPISODE)

        return sum(history.history['episode_reward'])/ sum(history.history['nb_steps'])