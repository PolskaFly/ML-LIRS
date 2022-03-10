"""
model: naive bayes
training data collection
1. re-access
2. demote from Rmax
time to predict: use start time parameter
feature:
1. flattened position 2. is hir before
"""

import sys
import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 64, DQN Code thanks to https://unnatsingh.medium.com/deep-q-network-with-pytorch-d1ca6f40bfda.
class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self, state_size,action_size, seed, fc1_unit=32,
                 fc2_unit = 32):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(QNetwork,self).__init__() ## calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size,fc1_unit)
        self.fc2 = nn.Linear(fc1_unit,fc2_unit)
        self.fc3 = nn.Linear(fc2_unit,action_size)
        
    def forward(self,x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

BUFFER_SIZE = int(2000)  #replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

class Agent():
    """Interacts with and learns form environment."""
    
    def __init__(self, state_size, action_size, seed, buffer):
        """Initialize an Agent object.
        
        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        
        #Q- Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=LR)
        
        # Replay memory 
        self.memory = ReplayBuffer(action_size, buffer,BATCH_SIZE,seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def step(self, state, action, reward, next_step, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step+1)% UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn

            if len(self.memory)>BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, GAMMA)
    def act(self, state, eps = 0):
        """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #Epsilon -greedy action selction
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_state, dones = experiences
        ## TODO: compute and minimize the loss
        criterion = torch.nn.MSELoss()
        # Local model is one which we need to train so it's in training mode
        self.qnetwork_local.train()
        # Target model is one with which we need to get our target so it's in evaluation mode
        # So that when we do a forward pass with target model it does not calculate gradient.
        # We will update target model weights with soft_update function
        self.qnetwork_target.eval()
        #shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.qnetwork_local(states).gather(1,actions)
    
        with torch.no_grad():
            labels_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (gamma* labels_next*(1-dones))
        
        loss = criterion(predicted_targets,labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local,self.qnetwork_target,TAU)
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
            
class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state",
                                                               "done"])
        self.seed = random.seed(seed)
        
    def add(self,state, action, reward, next_state,done):
        """Add a new experience to memory."""
        e = self.experiences(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory,k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Node:
    def __init__(self, b_num):
        self.block_number = b_num
        self.is_resident = False
        self.is_hir = True
        self.LIRS_prev = None
        self.LIRS_next = None
        self.HIR_prev = None
        self.HIR_next = None
        self.recency = False  # Rmax boundary
        self.recency0 = False  # LRU boundary
        self.dynamic_recency = False  # Non Resident Boundary
        self.refer_times = 0
        self.evicted_times = 0
        self.time_stamp = 0
        self.reuse_distance = 0
        self.is_last_hit = False
        """
        in dynamic, in lru, in lir stack, out lir stack 
        """
        self.position = [0, 0, 1]
        self.last_is_hir = True

        self.last_prediction = 0


class Trace:
    def __init__(self, t_name):
        self.trace_path = __location__ = os.path.join(__file__, "..", "..", "trace", t_name)
        self.parameter_path = __location__ = os.path.join(__file__, "..", "..", "cache_size", t_name)
        self.trace = []
        self.memory_size = []
        self.vm_size = -1
        self.trace_dict = dict()

    def get_trace(self):
        # print ("get trace from ", self.trace_path)
        with open(self.trace_path, "r") as f:
            for line in f.readlines():
                if not line == "*\n" and not line.strip() == "":
                    block_number = int(line)
                    self.trace.append(block_number)
                    self.trace_dict[block_number] = Node(block_number)
        self.vm_size = max(self.trace)
        return self.trace, self.trace_dict, len(self.trace)

    def get_parameter(self):
        # print ("get trace from ", self.parameter_path)
        with open(self.parameter_path, "r") as f:
            for line in f.readlines():
                if not line == "*\n":
                    self.memory_size.append(int(line.strip()))
        return self.memory_size

class WriteToFile:
    def __init__(self, tName, fp):
        self.tName = tName
        __location__ = os.path.join(__file__, "..", "..", "result_set", tName)
        self.FILE = open(__location__ + "\ml_lirs_v10_" + fp, "w+")

    def write_to_file(self, *args):
        data = ",".join(args)
        # Store the hit&miss ratio
        self.FILE.write(data + "\n")

class LIRS_Replace_Algorithm:
    def __init__(self, t_name, vm_size, trace_dict, mem_size, trace_size, model, start_use_model, mini_batch):
        self.trace_name = t_name
        # re-initialize trace_dict
        for k, v in trace_dict.items():
            trace_dict[k] = Node(k)

        self.model = model
        self.trace_size = trace_size
        self.page_table = trace_dict  # {block number : Node}
        self.MEMORY_SIZE = mem_size
        self.Free_Memory_Size = mem_size
        HIR_PERCENTAGE = 1.0
        MIN_HIR_MEMORY = 2
        self.HIR_SIZE = mem_size * (HIR_PERCENTAGE / 100)
        if self.HIR_SIZE < MIN_HIR_MEMORY:
            self.HIR_SIZE = MIN_HIR_MEMORY

        self.Stack_S_Head = None
        self.Sack_S_Tail = None
        self.Stack_Q_Head = None
        self.Stack_Q_Tail = None

        self.lir_size = 0
        self.Rmax = None
        self.Rmax0 = None
        self.inter_boundary = None
        self.inter_lost = True
        self.Rmax0_lost = True

        self.page_fault = 0
        self.temp_fault = 0
        self.page_hit = 0
        self.temp_hit = 0

        self.inter_hit_ratio = []
        self.dynamic_ratio = []
        self.lru_ratio = []
        self.virtual_time = []

        self.ratio_performance = []
        self.curr_performance = []

        self.prev_binary_performance = -2
        self.average_binary_performance = -2

        self.page_fault = 0
        self.page_hit = 0

        self.cache_size_pg_fault = 0
        self.cache_size_pg_hit = 0

        self.prevHitRate = 0
        self.broadPrevHitRate = 0

        self.last_ref_block = -1

        self.count_exampler = 0
        self.train = False

        self.predict_times = 0
        self.predict_H_L = 0
        self.predict_L_H = 0
        self.predict_H_H = 0
        self.predict_L_L = 0
        self.out_stack_hit = 0

        self.mini_batch_X = np.array([])
        self.mini_batch_X = np.array([])

        self.normalize_eviction = np.array([])

        self.start_use_model = start_use_model
        self.mini_batch = mini_batch

        # segment miss
        self.epoch = self.trace_size // 100
        self.last_miss = 0
        # self.seg_file = Segment_Miss(t_name, mem_size)
        self.positive_sampler = 0
        self.negative_sampler = 0

        self.good_decisions = 0
        self.bad_decisions = 0

        self.hit = False

        self.prevState = np.array([])

        self.agent = Agent(state_size=5,action_size=2,seed=0, buffer=self.MEMORY_SIZE * 2)


    def remove_stack_Q(self, b_num):
        if (not self.page_table[b_num].HIR_next and not self.page_table[b_num].HIR_prev):
            self.Stack_Q_Tail, self.Stack_Q_Head = None, None
        elif (self.page_table[b_num].HIR_next and self.page_table[b_num].HIR_prev):
            self.page_table[b_num].HIR_prev.HIR_next = self.page_table[b_num].HIR_next
            self.page_table[b_num].HIR_next.HIR_prev = self.page_table[b_num].HIR_prev
            self.page_table[b_num].HIR_next, self.page_table[b_num].HIR_prev = None, None
        elif (not self.page_table[b_num].HIR_prev):
            self.Stack_Q_Head = self.page_table[b_num].HIR_next
            self.Stack_Q_Head.HIR_prev.HIR_next = None
            self.Stack_Q_Head.HIR_prev = None
        elif (not self.page_table[b_num].HIR_next):
            self.Stack_Q_Tail = self.page_table[b_num].HIR_prev
            self.Stack_Q_Tail.HIR_next.HIR_prev = None
            self.Stack_Q_Tail.HIR_next = None
        else:
            raise ("Stack Q remove error \n")
        return True

    def add_stack_Q(self, b_num):
        if (not self.Stack_Q_Head):
            self.Stack_Q_Head = self.Stack_Q_Tail = self.page_table[b_num]
        else:
            self.page_table[b_num].HIR_next = self.Stack_Q_Head
            self.Stack_Q_Head.HIR_prev = self.page_table[b_num]
            self.Stack_Q_Head = self.page_table[b_num]

    def remove_stack_S(self, b_num):
        if (not self.page_table[b_num].LIRS_prev and not self.page_table[b_num].LIRS_next):
            return False

        if (self.page_table[b_num] == self.Rmax):
            self.Rmax = self.page_table[b_num].LIRS_prev
            self.find_new_Rmax()

        if (self.page_table[b_num] == self.Rmax0):
            self.Rmax0 = self.page_table[b_num].LIRS_prev

        if (self.page_table[b_num].LIRS_prev and self.page_table[b_num].LIRS_next):
            self.page_table[b_num].LIRS_prev.LIRS_next = self.page_table[b_num].LIRS_next
            self.page_table[b_num].LIRS_next.LIRS_prev = self.page_table[b_num].LIRS_prev
            self.page_table[b_num].LIRS_prev, self.page_table[b_num].LIRS_next = None, None
        elif (not self.page_table[b_num].LIRS_prev):
            self.Stack_S_Head = self.page_table[b_num].LIRS_next
            self.Stack_S_Head.LIRS_prev.LIRS_next = None
            self.Stack_S_Head.LIRS_prev = None
        elif (not self.page_table[b_num].LIRS_next):
            self.Stack_S_Tail = self.page_table[b_num].LIRS_prev
            self.Stack_S_Tail.LIRS_next.LIRS_prev = None
            self.Stack_S_Tail.LIRS_next = None
        else:
            raise ("Stack S remove error \n")
        return True

    def add_stack_S(self, b_num):
        if (not self.Stack_S_Head):
            self.Stack_S_Head = self.Stack_S_Tail = self.page_table[b_num]
            self.Rmax = self.page_table[b_num]
        else:
            self.page_table[b_num].LIRS_next = self.Stack_S_Head
            self.Stack_S_Head.LIRS_prev = self.page_table[b_num]
            self.Stack_S_Head = self.page_table[b_num]
        return True

    def find_new_Rmax(self):
        if (not self.Rmax):
            raise ("Warning Rmax \n")
        while self.Rmax.is_hir == True:
            self.Rmax.recency = False
            self.Rmax = self.Rmax.LIRS_prev

    # Prunes to a new Rmax0
    def find_new_Rmax0(self):
        if (not self.Rmax0):
            raise ("Warning Rmax0 \n")
        self.Rmax0.recency0 = False
        self.Rmax0 = self.Rmax0.LIRS_prev
        return self.Rmax0
    # Function that polls the hit ratio at different times in the trace
    def inter_ratios(self, v_time):
        if v_time % 250 == 0 and v_time != 0:
            h = (self.page_hit - self.temp_hit) / ((self.page_fault - self.temp_fault) +
                                                   (self.page_hit - self.temp_hit)) * 100
            self.inter_hit_ratio.append(float(h))
            self.temp_hit = self.page_hit
            self.temp_fault = self.page_fault

            self.virtual_time.append(v_time)

    def resident_number(self):
        count = 0
        for b_num, node in self.page_table.items():
            if (node.is_resident):
                count += 1
        print(self.MEMORY_SIZE, count)

    def collect_sampler(self, ref_block, *feature):
        self.count_exampler += 1

        features = []
        for f in feature:
            features += f

        position = [0, 0, 0]

        if (self.page_table[ref_block].recency0):
            position[0] = 1
        if (self.page_table[ref_block].recency):
            position[1] = 1
        if (not self.page_table[ref_block].recency):
            position[2] = 1
        self.page_table[ref_block].position = position

        return np.array(features)

    def print_information(self):
        print("======== Results ========")
        print("trace : ", self.trace_name)
        print("memory size : ", self.MEMORY_SIZE)
        print("trace_size : ", self.trace_size)
        print("Q size : ", self.HIR_SIZE)
        print("page hit : ", self.page_hit)
        print("page fault : ", self.page_fault)
        print("Hit ratio: ", self.page_hit / (self.page_fault + self.page_hit) * 100)
        print("Total: ", (self.page_fault + self.page_hit))
        print("Out stack hit : ", self.out_stack_hit)
        print("Predict Times =", self.predict_times, "H->L", self.predict_H_L, "L->H", self.predict_L_H)
        return self.MEMORY_SIZE, self.page_fault / (self.page_fault + self.page_hit) * 100, self.page_hit / (
                    self.page_fault + self.page_hit) * 100, self.inter_hit_ratio, self.dynamic_ratio, self.virtual_time

    def print_stack(self, v_time):
        if (v_time < 7995):
            return
        ptr = self.Stack_S_Head
        while (ptr):
            print("R" if ptr == self.Rmax0 else "", end="")
            print("H" if ptr.is_hir else "L", end="")
            print("r" if ptr.is_resident else "N", end="")
            print(ptr.block_number, end="-")
            ptr = ptr.LIRS_next
        print()

    def calculate_reward(self, prevHit, newHit):
        newCalc = newHit - prevHit

        if newCalc > 0:
            return newCalc * 200
        else:
            return newCalc * 100

    def LIRS(self, v_time, ref_block):
        self.hit = False

        if self.cache_size_pg_fault + self.cache_size_pg_hit == self.mini_batch:
            self.cache_size_pg_fault = self.cache_size_pg_fault/2
            self.cache_size_pg_hit = self.cache_size_pg_hit/2
            self.broadPrevHitRate = (self.page_hit/(self.page_hit+self.page_fault))*100

        self.inter_ratios(v_time)

        if not self.page_table[ref_block].recency:
            self.out_stack_hit += 1
            # print("out stack hit", v_time, ref_block)

        if (ref_block == self.last_ref_block):
            self.page_hit += 1
            self.cache_size_pg_hit += 1
            return

        self.last_ref_block = ref_block
        self.page_table[ref_block].refer_times += 1
        if (not self.page_table[ref_block].is_resident):
            self.page_fault += 1
            self.cache_size_pg_fault += 1

            if (self.Free_Memory_Size == 0):
                self.Stack_Q_Tail.is_resident = False
                self.Stack_Q_Tail.evicted_times += 1
                self.remove_stack_Q(self.Stack_Q_Tail.block_number)  # func(): remove block in the tail of stack Q
                self.Free_Memory_Size += 1
            elif (self.Free_Memory_Size > self.HIR_SIZE):
                self.page_table[ref_block].is_hir = False
                self.lir_size += 1

            self.Free_Memory_Size -= 1
        elif (self.page_table[ref_block].is_hir):
            self.remove_stack_Q(ref_block)  # func():

        if (self.page_table[ref_block].is_resident):
            self.page_hit += 1
            self.cache_size_pg_hit += 1
            self.hit = True

        # find new Rmax0
        if (self.Rmax0 and not self.page_table[ref_block].recency0):
            if (self.Rmax0 != self.page_table[ref_block]):
                self.find_new_Rmax0()

        self.page_table[ref_block].refer_times = ((self.page_table[ref_block].refer_times *
                                                  pow(.9, (v_time - self.page_table[ref_block].time_stamp))) + 1)


        # when use the data to predict the model
        if (self.count_exampler > self.start_use_model and self.Free_Memory_Size == 0):
            self.train = True

        self.remove_stack_S(ref_block)
        self.add_stack_S(ref_block)

        if (self.Free_Memory_Size == 0 and not self.Rmax0):
            self.Rmax0 = self.page_table[self.Rmax.block_number]
        

        self.page_table[ref_block].is_resident = True

        # start predict
        if (self.train):
            state = self.collect_sampler(ref_block, self.page_table[ref_block].position,
                                     [self.page_table[ref_block].last_is_hir,
                                      self.page_table[ref_block].refer_times])
            prediction = self.agent.act(state, .1)

            #if self.broadPrevHitRate > (self.page_hit/(self.page_hit+self.page_fault))*100:
            self.agent.step(self.prevState, prediction, self.calculate_reward(self.prevHitRate, (self.cache_size_pg_hit/(self.cache_size_pg_fault+self.cache_size_pg_hit))*100), state, False)

            self.page_table[ref_block].last_prediction = prediction

            self.predict_times += 1

            if (self.page_table[ref_block].is_hir):
                if (prediction == 0):
                    # H -> L
                    self.lir_size += 1
                    self.predict_H_L += 1
                    self.page_table[ref_block].is_hir = False
                    self.add_stack_Q(self.Rmax.block_number)  # func():
                    self.Rmax.is_hir = True
                    self.lir_size -= 1
                    self.Rmax.recency = False
                    self.find_new_Rmax()
                elif (prediction == 1):
                    # H -> H
                    self.predict_H_H += 1
                    self.add_stack_Q(ref_block)  # func():
            else:
                # print(prediction, self.lir_size, self.HIR_SIZE)
                # if prev LIR
                if (prediction == -1 and self.lir_size > self.HIR_SIZE):
                    # L -> H
                    self.predict_L_H += 1
                    self.lir_size -= 1
                    # print (v_time, ref_block, self.lir_size, "L -> H")
                    self.add_stack_Q(ref_block)  # func():
                    if (ref_block == self.Rmax.block_number):
                        self.Rmax.is_hir = True
                        self.Rmax.recency = False
                        self.find_new_Rmax()
                    else:
                        self.page_table[ref_block].is_hir = True

                elif (prediction == 1):
                    # L -> L do nothing
                    self.predict_L_L += 1
                    pass
        else:
            # origin lirs
            state = self.collect_sampler(ref_block, self.page_table[ref_block].position,
                                     [self.page_table[ref_block].last_is_hir,
                                      self.page_table[ref_block].refer_times])

            action = None
            if (self.page_table[ref_block].is_hir and self.page_table[ref_block].recency):
                self.page_table[ref_block].is_hir = False
                self.lir_size += 1

                self.page_table[ref_block].last_prediction = 1

                self.add_stack_Q(self.Rmax.block_number)  # func():
                self.Rmax.is_hir = True
                self.lir_size -= 1
                self.Rmax.recency = False

                self.find_new_Rmax()

                action = 0
            elif (self.page_table[ref_block].is_hir):
                self.page_table[ref_block].last_prediction = -1
                action = 1
                self.add_stack_Q(ref_block)  # func():

            if self.prevState.size > 0 and action != None:
                self.agent.step(self.prevState, action, self.calculate_reward(self.prevHitRate, (self.cache_size_pg_hit/(self.cache_size_pg_fault+self.cache_size_pg_hit)*100)), state, False)

        self.page_table[ref_block].recency = True
        self.page_table[ref_block].recency0 = True
        self.page_table[ref_block].last_is_hir = self.page_table[ref_block].is_hir
        self.page_table[ref_block].is_last_hit = self.hit
        self.page_table[ref_block].time_stamp = v_time

        self.prevState = state

        self.prevHitRate = (self.page_hit/(self.page_fault+self.page_hit)) * 100

def main(t_name, start_predict, mini_batch):
    # result file
    FILE = WriteToFile(t_name, t_name)
    # get trace
    trace_obj = Trace(t_name)
    # get the trace
    trace, trace_dict, trace_size = trace_obj.get_trace()
    memory_size = trace_obj.get_parameter()
    for memory in memory_size:
        RATIO_FILE = WriteToFile(t_name, str(memory) + "_" + t_name + "_ratios")
        model = MultinomialNB()

        lirs_replace = LIRS_Replace_Algorithm(t_name, trace_obj.vm_size, trace_dict, memory, trace_size, model,
                                              start_predict, mini_batch)
        for v_time, ref_block in enumerate(trace):
            lirs_replace.LIRS(v_time, ref_block)

        memory_size, miss_ratio, hit_ratio, inter_hit_ratio, dynamic_ratio, virtual_time = lirs_replace.print_information()
        FILE.write_to_file(str(memory_size), str(miss_ratio), str(hit_ratio))
        for i in range(len(inter_hit_ratio)):
            RATIO_FILE.write_to_file(str(virtual_time[i]), str(inter_hit_ratio[i]))


if __name__ == "__main__":
    # if (len(sys.argv) != 4):
    #     raise("usage: python3 XXX.py trace_name start_predict mini_batch")
    t_name = sys.argv[1]
    start_predict = int(sys.argv[2])
    mini_batch = int(sys.argv[3])
    main(t_name, start_predict, mini_batch)