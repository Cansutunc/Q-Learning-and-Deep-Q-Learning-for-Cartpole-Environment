import numpy as np
import random
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from collections import deque 
from tensorflow import gather_nd
from tensorflow.keras.losses import mean_squared_error 


class DeepQLearning:
    
    def __init__(self,env,gamma,epsilon,numberEpisodes):
        
    #-----------------1    
        self.env=env
        self.gamma=gamma #discount rate
        self.epsilon=epsilon #probability for exploring the environment
        self.numberEpisodes=numberEpisodes
        
        # state dimension
        self.stateDimension=4 #state vector consists of 4 entries, cart position cart velocity, pole angular position and pole angular veloctiy
        # action dimension
        self.actionDimension=2 # action 0 and action 1
        # this is the maximum size of the replay buffer
        self.replayBufferSize=300 #replay buffer stores all the tuples that have [current state,action,reward,next state,flagisterminal for checking next state]
        #300 means we can store 300 of tuples


        # this is the size of the training batch that is randomly sampled from the replay buffer
        self.batchReplayBufferSize=100
        
        # number of training episodes(100 in that case) it takes to update the target network parameters
        # that is, every updateTargetNetworkPeriod we update the target network parameters by copying from the online network
        self.updateTargetNetworkPeriod=100
        
        # this is the counter for updating the target network 
        # if this counter exceeds (updateTargetNetworkPeriod-1) we update the network 
        # parameters and reset the counter to zero, this process is repeated until the end of the training process
        self.counterUpdateTargetNetwork=0
        
        self.sumRewardsEpisode=[]
        
        # replay buffer 
        self.replayBuffer=deque(maxlen=self.replayBufferSize)
        
        # this is the main network
        # create network
        self.mainNetwork=self.createNetwork() #online network
        
        # this is the target network
        # create network
        #as you see target and online networks have the same structure
        self.targetNetwork=self.createNetwork()
        
        # copy the initial weights to targetNetwork from the mainnetwork(online network)
        self.targetNetwork.set_weights(self.mainNetwork.get_weights())
        
       #then we define an empty list of actions and it used to define cost functions
        # this list is used in the cost function to select certain entries of the 
        # predicted and true sample matrices in order to form the loss
        self.actionsAppend=[]
    

    # - loss - watch out here, this is a vector of (self.batchReplayBufferSize,1), 
    # with each entry being the squared error between the entries of y_true and y_pred
    # later on, the tensor flow will compute the scalar out of this vector (mean squared error) 
    
    #3-----------------------------------------------------
    def my_loss_fn(self,y_true, y_pred):
        
        s1,s2=y_true.shape
        #print(s1,s2)
        
        # this matrix defines indices of a set of entries that we want to 
        # extract from y_true and y_pred
        # s2=2
        # s1=self.batchReplayBufferSize
        indices=np.zeros(shape=(s1,s2))
        indices[:,0]=np.arange(s1)
        indices[:,1]=self.actionsAppend
        
        # gather_nd and mean_squared_error are TensorFlow functions
        loss = mean_squared_error(gather_nd(y_true,indices=indices.astype(int)), gather_nd(y_pred,indices=indices.astype(int)))
        #print(loss)
        return loss    

    
    # create a neural network
    #2-----------------------------------------------------

    def createNetwork(self):
        model=Sequential()
        model.add(Dense(128,input_dim=self.stateDimension,activation='relu'))
        model.add(Dense(56,activation='relu'))
        model.add(Dense(self.actionDimension,activation='linear'))
        # compile the network with the custom loss defined in my_loss_fn
        model.compile(optimizer = RMSprop(), loss = self.my_loss_fn, metrics = ['accuracy'])
        #it is important here we used custom cost funtion, loss = self.my_loss_fn
        return model
 
    #3----------------------------------------------------------------------
    def trainingEpisodes(self):
   
        
        # here we loop through the episodes
        for indexEpisode in range(self.numberEpisodes):
            
            # list that stores rewards per episode - this is necessary for keeping track of convergence 
            rewardsEpisode=[]
                       
            print("Simulating episode {}".format(indexEpisode))
            
            # reset the environment at the beginning of every episode
            (currentState,_)=self.env.reset()
                      
            # here we step from one state to another
            # this will loop until a terminal state is reached
            terminalState=False
            while not terminalState:
                                      
                # select an action on the basis of the current state, denoted by currentState
                action = self.selectAction(currentState,indexEpisode)
                
                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                (nextState, reward, terminalState,_,_) = self.env.step(action)          
                rewardsEpisode.append(reward)
         
                # add current state, action, reward, next state, and terminal flag to the replay buffer
                self.replayBuffer.append((currentState,action,reward,nextState,terminalState))
                
                # train network
                self.train_network()
                
                # set the current state for the next step
                currentState=nextState
            
            print("Sum of rewards {}".format(np.sum(rewardsEpisode)))        
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))

        

    def selectAction(self,state,index):
        import numpy as np
        
        # first index episodes we select completely random actions to have enough exploration
        # change this
        if index<1:
            return np.random.choice(self.actionDimension)   
            
        # Returns a random real number in the half-open interval [0.0, 1.0)
        # this number is used for the epsilon greedy approach
        randomNumber=np.random.random()
        
        # after index episodes, we slowly start to decrease the epsilon parameter
        if index>200:
            self.epsilon=0.999*self.epsilon
        
        # if this condition is satisfied, we are exploring, that is, we select random actions
        if randomNumber < self.epsilon:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(self.actionDimension, 1)[0]
            
        
        # otherwise, we are selecting greedy actions
        else:
            # we return the index where Qvalues[state,:] has the max value
            # that is, since the index denotes an action, we select greedy actions
                       
            Qvalues=self.mainNetwork.predict(state.reshape(1,4))
          
            return np.random.choice(np.where(Qvalues[0,:]==np.max(Qvalues[0,:]))[0])

    def train_network(self):
        if len(self.replayBuffer) > self.batchReplayBufferSize:
            random_sample_batch = random.sample(self.replayBuffer, self.batchReplayBufferSize)

            current_state_batch = np.zeros(shape=(self.batchReplayBufferSize, self.stateDimension))
            next_state_batch = np.zeros(shape=(self.batchReplayBufferSize, self.stateDimension))

            for index, (current_state, _, _, next_state, _) in enumerate(random_sample_batch):
                current_state_batch[index, :] = current_state
                next_state_batch[index, :] = next_state

            q_next_state_target_network = self.targetNetwork.predict(next_state_batch)
            q_current_state_main_network = self.mainNetwork.predict(current_state_batch)

            input_network = current_state_batch
            output_network = np.zeros(shape=(self.batchReplayBufferSize, self.actionDimension))

            actions_append = []

            for index, (current_state, action, reward, next_state, terminated) in enumerate(random_sample_batch):
                if terminated:
                    y = reward
                else:
                    alpha = 1.0 / (1.0 + self.visits[current_state][action])

                    expected_q = np.max(q_next_state_target_network[index])

                    y = (1 - alpha) * q_current_state_main_network[index, action] + alpha * (
                        reward + self.gamma * expected_q
                    )

                    self.visits[current_state][action] += 1

                actions_append.append(action)
                output_network[index] = q_current_state_main_network[index]
                output_network[index, action] = y

            self.mainNetwork.fit(input_network, output_network, batch_size=self.batchReplayBufferSize, verbose=0, epochs=100)

            self.counterUpdateTargetNetwork += 1
            if self.counterUpdateTargetNetwork > (self.updateTargetNetworkPeriod - 1):
                self.targetNetwork.set_weights(self.mainNetwork.get_weights())
                print("Target network updated!")
                print("Counter value {}".format(self.counterUpdateTargetNetwork))
                self.counterUpdateTargetNetwork = 0

                    
 