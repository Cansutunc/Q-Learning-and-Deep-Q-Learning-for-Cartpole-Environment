
import gym
import numpy as np
import time
import matplotlib.pyplot as plt
from functions import NonDeterministicQLearning

# Classical gym
env = gym.make('CartPole-v1')
(state, _) = env.reset()

# Define state discretization parameters
upper_bounds = env.observation_space.high
lower_bounds = env.observation_space.low
cart_velocity_min = -3
cart_velocity_max = 3
pole_angle_velocity_min = -10
pole_angle_velocity_max = 10
upper_bounds[1] = cart_velocity_max
upper_bounds[3] = pole_angle_velocity_max
lower_bounds[1] = cart_velocity_min
lower_bounds[3] = pole_angle_velocity_min

#values for discritezation
number_of_bins_position = 30
number_of_bins_velocity = 30
number_of_bins_angle = 30
number_of_bins_angle_velocity = 30
number_of_bins = [number_of_bins_position, number_of_bins_velocity, number_of_bins_angle, number_of_bins_angle_velocity]

# Define parameters
alpha = 0.1 #step size
gamma = 1   #discount rate 
epsilon = 0.2       #greedy
number_episodes = 1000

# Create an object using NonDeterministicQLearning
Q2 = NonDeterministicQLearning(env, alpha, gamma, epsilon, number_episodes, number_of_bins, lower_bounds, upper_bounds)

# Run the Q-Learning algorithm
Q2.simulateEpisodes()

# Simulate the learned strategy
(obtained_rewards_optimal, env1) = Q2.simulateLearnedStrategy()

plt.figure(figsize=(12, 5))
plt.plot(Q2.sumRewardsEpisode, color='blue', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Sum of Rewards in Episode')
plt.yscale('log')
plt.savefig('convergence.png')
plt.show()

# Close the environment
env1.close()

# Get the sum of rewards
np.sum(obtained_rewards_optimal)

# Now simulate a random strategy
(obtained_rewards_random, env2) = Q2.simulateRandomStrategy()

plt.hist(obtained_rewards_random)
plt.xlabel('Sum of Rewards')
plt.ylabel('Percentage')
plt.savefig('histogram.png')
plt.show()

# Run this several times and compare with a random learning strategy
(obtained_rewards_optimal, env1) = Q2.simulateLearnedStrategy()
