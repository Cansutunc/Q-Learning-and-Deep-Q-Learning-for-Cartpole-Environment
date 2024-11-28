
# import the class
from functions import DeepQLearning
# classical gym 
import gym
# instead of gym, import gymnasium 
#import gymnasium as gym

# create environment
env=gym.make('CartPole-v1')

# select the parameters
gamma=1
# probability parameter for the epsilon-greedy approach
epsilon=0.1
# number of training episodes
numberEpisodes=2000

# create an object
LearningQDeep=DeepQLearning(env,gamma,epsilon,numberEpisodes)
# run the learning process
LearningQDeep.trainingEpisodes()
# get the obtained rewards in every episode
LearningQDeep.sumRewardsEpisode

#  summarize the model
LearningQDeep.mainNetwork.summary()
# save the model, this is important, since it takes long time to train the model 
# and we will need model in another file to visualize the trained model performance
LearningQDeep.mainNetwork.save("trained_model_temp.h5")



