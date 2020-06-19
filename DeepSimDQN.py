import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable




ACTIONS = 2    # total available action number for the game: 0 for Danzig
               # 1 for steepest


class DeepSimDQN(nn.Module):
    # empty_state = np.zeros((128, 72), dtype=np.float32)
    # state is the reduced cost and the objective value
    def __init__(self, epsilon, mem_size, dim):
        """Initialization

           epsilon: initial epsilon for exploration
                   mem_size: memory size for experience replay
                   cuda: use cuda or not
        """
        dim = dim.split(',')
        super(DeepSimDQN, self).__init__()
        self.train = None
        # init replay memory
        self.replay_memory = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = epsilon
        self.actions = ACTIONS
        self.mem_size = mem_size
        self.L = int(dim[0])
        self.P = int(dim[1])
        self.k = int(dim[2])
        self.p = float(dim[3])
        # init Q network
        self.createQNetwork()


    def createQNetwork(self):
        """ Create dqn, invoked by `__init__`

            model structure: 8 fc
            change it to your new design
        """
        self.fc1 = nn.Linear(self.P+2-self.k, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.actions)
        self.tanh = nn.Tanh()

    def get_q_value(self, o):
        """Get Q value estimation w.r.t. current observation `o`

           o -- current observation
        """
        # get Q estimation
        out = self.fc1(o)
        out = self.relu(out)
        for i in range(6):
            out = self.fc2(out)
            out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

    def forward(self, o):
        """Forward procedure to get MSE loss

           o -- current observation
        """
        # get Q(s,a;\theta)
        q = self.get_q_value(o)
        return q

    def set_train(self):
        """Set phase TRAIN
        """
        self.train = True

    def set_eval(self):
        """Set phase EVALUATION
        """
        self.train = False

    def set_initial_state(self, state=None):
        """Set initial state

           state: initial state. if None, use `BrainDQN.empty_state`
        """
        if state is None:
            self.current_state = np.zeros(self.P+self.L*2+2)
        else:
            self.current_state = state


    def store_transition(self, o_next, action, reward, terminal):
        """Store transition (\fan_t, a_t, r_t, \fan_{t+1})

           o_next: next observation, \fan_{t+1}
           action: action, a_t
           reward: reward, r_t
           terminal: terminal(\fan_{t+1})
        """
        # next_state = np.append(self.current_state[1:,:,:], o_next.reshape((1,)+o_next.shape), axis=0)
        self.replay_memory.append((self.current_state, action, reward, o_next, terminal))
        if len(self.replay_memory) > self.mem_size:
            self.replay_memory.popleft()
        self.current_state = o_next

    def get_action_randomly(self):
        """Get action randomly
        """
        # action = np.zeros(self.actions, dtype=np.float32)
        #action_index = random.randrange(self.actions)
        action_index = 0 if random.random() < 0.5 else 1
        # action[action_index] = 1
        return action_index

    def get_optim_action(self):
        """Get optimal action based on current state
        """
        state = self.current_state
        with torch.no_grad():
            state_var = torch.from_numpy(state)
            state_var = state_var.float()
            q_value = self.forward(state_var)
        action_index = q_value.argmax()
        # action_index = action_index[0]
        # action = np.zeros(self.actions, dtype=np.float32)
        # action[action_index] = 1
        return action_index

    def get_action(self):
        """Get action w.r.t current state
        """
        if self.train and random.random() <= self.epsilon:
            return self.get_action_randomly()
        return self.get_optim_action()

    def increase_time_step(self, time_step=1):
        """increase time step"""
        self.time_step += time_step