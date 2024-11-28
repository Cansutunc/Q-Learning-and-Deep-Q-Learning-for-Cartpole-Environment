
import numpy as np

class NonDeterministicQLearning:
  
    def __init__(self, env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds):
        import numpy as np
        
        self.env = env
        self.alpha = alpha
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.actionNumber = env.action_space.n 
        self.numberEpisodes = numberEpisodes
        self.numberOfBins = numberOfBins
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        
        # List to store sum of rewards in every learning episode
        self.sumRewardsEpisode = []
        
        # Action value function matrix 
        self.Qmatrix = np.random.uniform(low=0, high=1, size=(numberOfBins[0], numberOfBins[1], numberOfBins[2], numberOfBins[3], self.actionNumber))
        
        # Dictionary to store visit counts
        self.visits = dict()

    def returnIndexState(self, state):
        position, velocity, angle, angularVelocity = state
        
        cartPositionBin = np.linspace(self.lowerBounds[0], self.upperBounds[0], self.numberOfBins[0])
        cartVelocityBin = np.linspace(self.lowerBounds[1], self.upperBounds[1], self.numberOfBins[1])
        poleAngleBin = np.linspace(self.lowerBounds[2], self.upperBounds[2], self.numberOfBins[2])
        poleAngleVelocityBin = np.linspace(self.lowerBounds[3], self.upperBounds[3], self.numberOfBins[3])
        
        indexPosition = np.maximum(np.digitize(position, cartPositionBin) - 1, 0)
        indexVelocity = np.maximum(np.digitize(velocity, cartVelocityBin) - 1, 0)
        indexAngle = np.maximum(np.digitize(angle, poleAngleBin) - 1, 0)
        indexAngularVelocity = np.maximum(np.digitize(angularVelocity, poleAngleVelocityBin) - 1, 0)
        
        return tuple([indexPosition, indexVelocity, indexAngle, indexAngularVelocity])

    def selectAction(self, state, index):
        if index < 500:
            return np.random.choice(self.actionNumber)   
            
        randomNumber = np.random.random()
        
        if index > 700:
            self.epsilon = 0.999 * self.epsilon
        
        if randomNumber < self.epsilon:
            return np.random.choice(self.actionNumber)            
        else:
            return np.random.choice(np.where(self.Qmatrix[self.returnIndexState(state)] == np.max(self.Qmatrix[self.returnIndexState(state)]))[0])

   
    def update_q_value(self, state, action, reward, next_state, terminal_state):
        state_index = self.returnIndexState(state)
        action_index = action

        if state_index not in self.visits:
            self.visits[state_index] = dict()

        if action_index not in self.visits[state_index]:
            self.visits[state_index][action_index] = 0

        # Calculate alpha based on the number of visits
        alpha = 1 / (1 + self.visits[state_index][action_index])

        next_state_index = self.returnIndexState(next_state)
        Qmax_prime = np.max(self.Qmatrix[next_state_index])

        # Update the visit count
        self.visits[state_index][action_index] += 1
        
        # Update Q-value using the modified non-deterministic Q-learning update rule
        self.Qmatrix[state_index + (action_index,)] = (1 - alpha) * self.Qmatrix[state_index + (action_index,)] + \
            alpha * (reward + self.gamma * Qmax_prime)

    def simulateEpisodes(self):
        import numpy as np
        
        for indexEpisode in range(self.numberEpisodes):
            rewards_episode = []
            
            (state, _) = self.env.reset()
            state = list(state)
          
            print("Simulating episode {}".format(indexEpisode))
            
            terminal_state = False
            while not terminal_state:
                state_index = self.returnIndexState(state)
                action = self.selectAction(state, indexEpisode)
                
                (next_state, reward, terminal_state, _, _) = self.env.step(action)          
                
                rewards_episode.append(reward)
                
                next_state = list(next_state)
                
                self.update_q_value(state, action, reward, next_state, terminal_state)
                
                state = next_state

            print("Sum of rewards {}".format(np.sum(rewards_episode)))        
            self.sumRewardsEpisode.append(np.sum(rewards_episode))

    def simulateLearnedStrategy(self):
        import gym 
        import time
        env1 = gym.make('CartPole-v1', render_mode='human')
        (current_state, _) = env1.reset()
        env1.render()
        time_steps = 100
        obtained_rewards = []
        
        for time_index in range(time_steps):
            action_in_state = np.random.choice(np.where(self.Qmatrix[self.returnIndexState(current_state)] == np.max(self.Qmatrix[self.returnIndexState(current_state)]))[0])
            current_state, reward, terminated, truncated, info = env1.step(action_in_state)
            obtained_rewards.append(reward)   
            time.sleep(0.05)
            if terminated:
                time.sleep(1)
                break
        return obtained_rewards, env1

    def simulateRandomStrategy(self):
        import gym 
        import time
        import numpy as np
        env2 = gym.make('CartPole-v1')
        (current_state, _) = env2.reset()
        env2.render()
        episode_number = 100
        time_steps = 100
        sum_rewards_episodes = []
        
        for episode_index in range(episode_number):
            rewards_single_episode = []
            initial_state = env2.reset()
            print(episode_index)
            for time_index in range(time_steps):
                random_action = env2.action_space.sample()
                observation, reward, terminated, truncated, info = env2.step(random_action)
                rewards_single_episode.append(reward)
                if terminated:
                    break      
            sum_rewards_episodes.append(np.sum(rewards_single_episode))
        return sum_rewards_episodes, env2



               
     
                
            
            
            
            
        
        
        
        
        
        
                
                
                
                
                
                
                
                
                
            
            
            
            
        
        
        
        
        
        
