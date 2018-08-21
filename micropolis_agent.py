# Notebook from https://github.com/yandexdataschool/AgentNet/blob/master/examples/Playing%20Atari%20with%20Deep%20Reinforcement%20Learning%20%28OpenAI%20Gym%29.ipynb 

#setup theano/lasagne. Prefer GPU
# env THEANO_FLAGS=device=gpu0,floatX=float32
import gym_micropolis
import gym.spaces
import theano, lasagne
from lasagne.layers import *
#If you are running on a server, launch xvfb to record game videos
#Please make sure you have xvfb installed
import os
if os.environ.get("DISPLAY") is str and len(os.environ.get("DISPLAY"))!=0:
    pass
   # !bash ../xvfb start
   # %env DISPLAY=:1

import matplotlib.pyplot as plt
import numpy as np
# matplotlib inline

#number of parallel agents and batch sequence length (measures in actions, since game time is independent)
N_AGENTS = 1
SEQ_LENGTH = 3

import gym
#game maker consider https://gym.openai.com/envs
def make_env():
    env = gym.make("MicropolisEnv-v0")
    return env

#spawn game instance
env = make_env()
MAP_X = env.MAP_X
MAP_Y = env.MAP_Y
num_tools = env.num_tools
observation_shape = env.observation_space.shape
num_zones = observation_shape[0]
n_actions = env.num_tools * MAP_X * MAP_Y
env.close()
del(env)

# how many steps to unroll for recurrence
num_steps = 1
#observation
observation_layer = InputLayer((None,) + observation_shape,)
obs_layer_shape = get_output_shape(observation_layer)
print(obs_layer_shape)
obs_t = reshape(observation_layer, (-1, 1,) + observation_shape,) 

from lasagne.nonlinearities import elu,tanh,softmax

#network body
in_conv0 = InputLayer(get_output_shape(observation_layer))
# suppose this convolution maintains binary zone info, and adds binary info about road connectivity, power, and desirability
conv0 = Conv2DLayer(in_conv0, num_zones + 3, 1, stride=1, nonlinearity=elu, pad='same')
print(get_output_shape(conv0))

in_conv_rec = InputLayer(get_output_shape(conv0))
# suppose this recurrent convolution allows for, e.g., spread of power across network
conv_rec = Conv2DLayer(in_conv_rec, num_zones + 3, 3, stride=1, nonlinearity=elu, pad='same')
print(get_output_shape(conv_rec))

recur = CustomRecurrentLayer(obs_t, conv0, conv_rec)
print(get_output_shape(recur))
recur_single = reshape(recur, (([0], num_zones + 3,) + (MAP_X, MAP_Y,)))



#conv1 = Conv2DLayer(recur_single,16,3,stride=1,nonlinearity=tanh, pad='same')
#print(get_output_shape(conv1))
#
#conv2 = Conv2DLayer(conv1,32,3,stride=1,nonlinearity=elu, pad='same')
#print(get_output_shape(conv2))
#
#conv3 = Conv2DLayer(conv2,64,3,stride=1,nonlinearity=tanh, pad='same')
#print(get_output_shape(conv3))
#
conv4 = Conv2DLayer(recur_single, num_tools, 3, stride=1, nonlinearity=elu, pad='same')
print(get_output_shape(conv4))

# intended a coordinate-invariant dense layer - each filter is the size of the
# map
conv_mapwide = Conv2DLayer(conv4, num_tools, MAP_X + 1, stride=1, nonlinearity=elu, pad="same")
print(get_output_shape(conv_mapwide))

qvalues_layer = reshape(conv_mapwide, ([0],-1))
print(get_output_shape(qvalues_layer))

#baseline for all qvalues
#qvalues_layer = DenseLayer(dense,n_actions,nonlinearity=None,name='qval')
#sample actions proportionally to policy_layer
from agentnet.resolver import EpsilonGreedyResolver
action_layer = EpsilonGreedyResolver(qvalues_layer)

from agentnet.target_network import TargetNetwork
targetnet = TargetNetwork(qvalues_layer,recur_single)
qvalues_old = targetnet.output_layers
from agentnet.agent import Agent
#all together
agent = Agent(observation_layers=observation_layer,
               policy_estimators=(qvalues_layer, qvalues_old),
              #agent_states={dense:dense0},
              action_layers=action_layer)

#Since it's a single lasagne network, one can get it's weights, output, etc
weights = lasagne.layers.get_all_params(qvalues_layer,trainable=True)
weights



from agentnet.experiments.openai_gym.pool import EnvPool

pool = EnvPool(agent,make_env, N_AGENTS)


## %%time
##interact for 7 ticks
#_,action_log,reward_log,_,_,_  = pool.interact(10)
#
#print('actions:')
#print(action_log[0])
#print("rewards")
#print(reward_log[0])


#load first sessions (this function calls interact and remembers sessions)
pool.update(SEQ_LENGTH)

#get agent's Qvalues obtained via experience replay
replay = pool.experience_replay

_,_,_,_,(qvalues_seq,old_qvalues_seq) = agent.get_sessions(
    replay,
    session_length=SEQ_LENGTH,
    experience_replay=True,
)

#get reference Qvalues according to Qlearning algorithm
from agentnet.learning import qlearning

#crop rewards to [-1,+1] to avoid explosion.
rewards = replay.rewards/10.

#loss for Qlearning = 
#(Q(s,a) - (r+ gamma*r' + gamma^2*r'' + ...  +gamma^10*Q(s_{t+10},a_max)))^2
elwise_mse_loss = qlearning.get_elementwise_objective(qvalues_seq,
                                                      replay.actions[0],
                                                      rewards,
                                                      replay.is_alive,
                                                      qvalues_target=old_qvalues_seq,
                                                      gamma_or_gammas=0.99,
                                                      n_steps=10)

#mean over all batches and time ticks
loss = elwise_mse_loss.mean()
# Compute weight updates
updates = lasagne.updates.adam(loss,weights,learning_rate=1e-4)
#compile train function
train_step = theano.function([],loss,updates=updates)

action_layer.epsilon.set_value(0)
untrained_reward = np.mean(pool.evaluate(save_path="./records", record_video=False, use_monitor=False, t_max=1))

epoch_counter = 1

#full game rewards
rewards = {}
loss,reward_per_tick,reward =0,0,0

from tqdm import trange
#from IPython.display import clear_output

for i in trange(150000):    
    ##update agent's epsilon (in e-greedy policy)
    current_epsilon = 0.01 + 0.45*np.exp(-epoch_counter/20000.)
    action_layer.epsilon.set_value(np.float32(current_epsilon))

    #play
    pool.update(SEQ_LENGTH)

    #train
    loss = 0.95*loss + 0.05*train_step()
    targetnet.load_weights(0.01)
    
    
    if epoch_counter%10==0:
        #average reward per game tick in current experience replay pool
        reward_per_tick = 0.95*reward_per_tick + 0.05*pool.experience_replay.rewards.get_value().mean()
        print("iter=%i\tepsilon=%.3f\tloss=%.3f\treward/tick=%.3f"%(epoch_counter,
                                                               current_epsilon,
                                                               loss,
                                                               reward_per_tick))
        
    ##record current learning progress and show learning curves
    if epoch_counter%100 ==0:
        action_layer.epsilon.set_value(0)
        reward = 0.95*reward + 0.05*np.mean(pool.evaluate(record_video=False, use_monitor=False, t_max = 20))
        action_layer.epsilon.set_value(np.float32(current_epsilon))
        
        rewards[epoch_counter] = reward
        
        plt.plot(*zip(*sorted(rewards.items(),key=lambda t :t)))
      # plt.show(block = False)
        

    
    epoch_counter  +=1

    
# Time to drink some coffee (or smoke something)!


import pandas as pd
plt.plot(*zip(*sorted(rewards.items(),key=lambda k:k[0])))

from agentnet.utils.persistence import save
save(action_layer,"pacman.pcl")

###LOAD FROM HERE
from agentnet.utils.persistence import load
load(action_layer,"pacman.pcl")


action_layer.epsilon.set_value(0.01)
rw = pool.evaluate(n_games=20,save_path="./records",record_video=False)
print("mean session score=%f.5"%np.mean(rw))


##show video
#from IPython.display import HTML
#import os
#
#video_names = list(filter(lambda s:s.endswith(".mp4"),os.listdir("./records/")))
#
#HTML("""
#<video width="640" height="480" controls>
#  <source src="{}" type="video/mp4">
#</video>
#""".format("./videos/"+video_names[-1])) #this may or may not be _last_ video. Try other indices
