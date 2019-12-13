import gym
from RLzoo.common.utils import make_env, set_seed
from RLZoo.algorithms.ac.ac import AC
from RLzoo.common.value_networks import ValueNetwork
from RLzoo.common.policy_networks import StochasticPolicyNetwork

''' load environment '''
env = gym.make('CartPole-v0').unwrapped
obs_space = env.observation_space
act_space = env.action_space
# reproducible
seed = 2
set_seed(seed, env)

''' build networks for the algorithm '''
num_hidden_layer = 4 #number of hidden layers for the networks
hidden_dim = 64 # dimension of hidden layers for the networks
with tf.name_scope('AC'):
        with tf.name_scope('Critic'):
            	# choose the critic network, can be replaced with customized network
                critic = ValueNetwork(obs_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
        with tf.name_scope('Actor'):
            	# choose the actor network, can be replaced with customized network
                actor = StochasticPolicyNetwork(obs_space, act_space, hidden_dim_list=num_hidden_layer * [hidden_dim], output_activation=tf.nn.tanh)
net_list = [actor, critic] # list of the networks

''' choose optimizers '''
a_lr, c_lr = 1e-4, 1e-2  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
a_optimizer = tf.optimizers.Adam(a_lr)
c_optimizer = tf.optimizers.Adam(c_lr)
optimizers_list=[a_optimizer, c_optimizer]  # list of optimizers

# intialize the algorithm model, with algorithm parameters passed in
model = AC(net_list, optimizers_list)
''' 
full list of arguments for the algorithm
----------------------------------------
net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
optimizers_list: a list of optimizers for all networks and differentiable variables
gamma: discounted factor of reward
action_range: scale of action values
'''

# start the training process, with learning parameters passed in
model.learn(env, train_episodes=500,  max_steps=200,
            save_interval=50, mode='train', render=False)
''' 
full list of parameters for training
---------------------------------------
env: learning environment
train_episodes:  total number of episodes for training
test_episodes:  total number of episodes for testing
max_steps:  maximum number of steps for one episode
save_interval: time steps for saving the weights and plotting the results
mode: 'train' or 'test'
render:  if true, visualize the environment
'''

# test after training
model.learn(env, test_episodes=100, max_steps=200,  mode='test', render=True)