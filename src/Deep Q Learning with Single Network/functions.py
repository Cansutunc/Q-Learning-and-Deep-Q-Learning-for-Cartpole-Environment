import numpy as np
import random
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from collections import deque 
from tensorflow.keras.losses import MeanSquaredError

class DeepQLearning:
    
    def __init__(self, env, gamma, epsilon, number_episodes):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.number_episodes = number_episodes
        
        # state dimension
        self.state_dimension = 4
        # action dimension
        self.action_dimension = 2
        # this is the maximum size of the replay buffer
        self.replay_buffer_size = 300
        # this is the size of the training batch that is randomly sampled from the replay buffer
        self.batch_replay_buffer_size = 50
        # number of training episodes it takes to update the network parameters
        self.update_network_period = 10
        # counter for updating the network parameters
        self.counter_update_network = 0
        self.sum_rewards_episode = []
        # replay buffer 
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        # visits dictionary
        self.visits = {}
        
        # create main network
        self.main_network = self.create_network()

    # create a neural network
    def create_network(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_dimension, activation='relu'))
        model.add(Dense(56, activation='relu'))
        model.add(Dense(self.action_dimension, activation='linear'))
        model.compile(optimizer=RMSprop(), loss=MeanSquaredError(), metrics=['accuracy'])
        return model

    def trainingEpisodes(self):
        # loop through the episodes
        for index_episode in range(self.number_episodes):
            # list that stores rewards per episode - this is necessary for keeping track of convergence 
            rewards_episode = []
            print("Simulating episode {}".format(index_episode))
            
            # reset the environment at the beginning of every episode
            current_state, _ = self.env.reset()
            current_state = np.reshape(current_state, [1, self.state_dimension])
            
            # here we step from one state to another
            # this will loop until a terminal state is reached
            terminal_state = False
            
            while not terminal_state:
                # select an action on the basis of the current state, denoted by current_state
                action = self.select_action(current_state, index_episode)
                
                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                next_state, reward, terminal_state, _, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_dimension])
                
                rewards_episode.append(reward)
                
                # add current state, action, reward, next state, and terminal flag to the replay buffer
                self.replay_buffer.append((current_state, action, reward, next_state, terminal_state))
                
                # train network
                self.train_network()
                
                # set the current state for the next step
                current_state = next_state
            
            print("Sum of rewards {}".format(np.sum(rewards_episode)))        
            self.sum_rewards_episode.append(np.sum(rewards_episode))

    def select_action(self, state, index):
        # first index episodes we select completely random actions to have enough exploration
        if index < 1:
            return np.random.choice(self.action_dimension)   
            
        # Returns a random real number in the half-open interval [0.0, 1.0)
        # this number is used for the epsilon-greedy approach
        random_number = np.random.random()
        
        # after index episodes, we slowly start to decrease the epsilon parameter
        if index > 200:
            self.epsilon = 0.999 * self.epsilon
        
        # if this condition is satisfied, we are exploring, that is, we select random actions
        if random_number < self.epsilon:
            return np.random.choice(self.action_dimension, 1)[0]
            
        # otherwise, we are selecting greedy actions
        else:
            # we return the index where Qvalues[state,:] has the max value
            # that is, since the index denotes an action, we select greedy actions
            q_values = self.main_network.predict(state)
            return np.random.choice(np.where(q_values[0, :] == np.max(q_values[0, :]))[0])

    def train_network(self):
        #This condition checks whether the replay buffer has accumulated enough experiences to create a batch for training.
        if len(self.replay_buffer) > self.batch_replay_buffer_size:
            #Randomly samples batch_replay_buffer_size experiences from the replay buffer. 
            random_sample_batch = random.sample(self.replay_buffer, self.batch_replay_buffer_size)
            
            #Initializes arrays to store the current state and next state for each experience in the batch.
            current_state_batch = np.zeros(shape=(self.batch_replay_buffer_size, self.state_dimension))
            next_state_batch = np.zeros(shape=(self.batch_replay_buffer_size, self.state_dimension))

            for index, (current_state, _, _, next_state, _) in enumerate(random_sample_batch):
                current_state_batch[index, :] = current_state.flatten()
                next_state_batch[index, :] = next_state.flatten()

            #Uses the neural network (main_network) to predict Q-values for the next states and current states in the batch.
            q_next_state_main_network = self.main_network.predict(next_state_batch)
            q_current_state_main_network = self.main_network.predict(current_state_batch)

            #Initializes variables to store input states (input_network), output Q-values (output_network), and a list to append the chosen actions (actions_append) for each experience in the batch.
            input_network = current_state_batch
            output_network = np.zeros(shape=(self.batch_replay_buffer_size, self.action_dimension))

            actions_append = []

            #Iterates through the sampled batch, calculating updated Q-values (y) based on the Q-learning update rule. 
            for index, (current_state, action, reward, next_state, terminated) in enumerate(random_sample_batch):
                if terminated:
                    y = reward
                else:
                    alpha = 1.0 / (1.0 + self.visits.get(tuple(current_state.flatten()), {}).get(action, 0))
                    expected_q = np.max(q_next_state_main_network[index])
                    y = (1 - alpha) * q_current_state_main_network[index, action] + alpha * (reward + self.gamma * expected_q)
                    if tuple(current_state.flatten()) not in self.visits:
                        self.visits[tuple(current_state.flatten())] = {}
                    self.visits[tuple(current_state.flatten())][action] = self.visits[tuple(current_state.flatten())].get(action, 0) + 1

                actions_append.append(action)
                output_network[index] = q_current_state_main_network[index]
                output_network[index, action] = y

            #It also updates the neural network (main_network) using the input and output networks with a fit operation.
            self.main_network.fit(input_network, output_network, batch_size=self.batch_replay_buffer_size, verbose=0, epochs=100)

            #Increments a counter (counter_update_network) to keep track of the number of updates. If the counter reaches the specified update_network_period, it prints a message indicating that the network parameters have been updated.
            self.counter_update_network += 1
            if self.counter_update_network > (self.update_network_period - 1):
                print("Network parameters updated!")
                print("Counter value {}".format(self.counter_update_network))
                self.counter_update_network = 0




