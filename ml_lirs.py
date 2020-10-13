import os
import sys
from collections import deque
from collections import OrderedDict
import codecs
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from statistics import median
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

class Block:
    def __init__(self, Block_Number, is_resident, is_hir_block, in_stack, reuse_distance, min_b_range, max_b_range, median):
        self.b_num = Block_Number
        self.is_resident = is_resident
        self.is_hir = is_hir_block
        self.in_stack = in_stack
        self.reuse_distance = reuse_distance
        self.min_b_range = min_b_range
        self.max_b_range = max_b_range
        self.median = median


HIR_PERCENTAGE = 1.0
MIN_HIR_MEMORY = 2

class LIRS:
    def __init__(self, trace, mem, result, info):
        self.trace = trace
        # Init the queue
        self.pg_table = deque()
        for x in range(vm_size + 1):
            # [Block Number, is_resident, is_hir_block, in_stack]
            block = Block(x, False, True, False, -1, 0, 0, 0)
            self.pg_table.append(block)

        self.HIR_SIZE = mem * (HIR_PERCENTAGE / 100)
        if self.HIR_SIZE < MIN_HIR_MEMORY:
            self.HIR_SIZE = MIN_HIR_MEMORY
        self.BLOCK_RANGE = self.HIR_SIZE * 5

        if self.BLOCK_RANGE > 15:
            self.BLOCK_RANGE = 15
        # Init End

        #Creating stacks and lists
        self.lir_stack = OrderedDict()
        self.hir_stack = OrderedDict()
        self.eviction_list = []
        self.pg_hits = 0
        self.pg_faults = 0
        self.free_mem = mem
        self.mem = mem
        self.mRatio = 0
        self.in_stack_miss = 0
        self.out_stack_hit = 0
        self.out_stack_miss = 0
        self.lir_size = 0

        self.block_range_window = deque()

        self.result = result
        self.info = info


    def find_lru(self, s: OrderedDict, pg_table: deque):
        temp_s = list(s)
        # Always get first item of the temp_s
        while self.pg_table[temp_s[0]].is_hir:
            self.pg_table[temp_s[0]].in_stack = False
            s.popitem(last=False)
            del temp_s[0]

    def get_range(self, trace) -> int:
        max = 0
        for x in trace:
            if x > max:
                max = x
        return max

    def print_information(self, mem, hits, faults, size):
        print("Memory Size: ", mem)
        print("Hits: ", hits)
        print("Faults: ", faults)
        print("Total: ", hits + faults)
        print("Hit Ratio: ", hits / (hits + faults) * 100)

    def replace_lir_block(self, pg_table, lir_size):
        temp_block = self.lir_stack.popitem(last=False)
        self.pg_table[temp_block[1]].is_hir = True
        self.pg_table[temp_block[1]].in_stack = False
        self.hir_stack[temp_block[1]] = temp_block[1]
        self.find_lru(self.lir_stack, pg_table)
        self.lir_size -= 1
        return self.lir_size

    def ML_LIRS_Replace_Algorithm(self):
        # Creating variables
        last_ref_block = -1
        trained = False
        training_data = deque()
        label_data = deque()
        train_amount = 100
        count_nonevict = 0
        count_evict = 0

        model = GradientBoostingClassifier(n_estimators=30, learning_rate=.1, max_features=2, max_depth=15, random_state=0)

        for i in range(len(self.trace)):
            ref_block = self.trace[i]

            # setting block range
            if len(self.block_range_window) == self.BLOCK_RANGE:
                self.pg_table[ref_block].min_b_range = min(self.block_range_window)
                self.pg_table[ref_block].max_b_range = max(self.block_range_window)
                self.pg_table[ref_block].median = median(self.block_range_window)
            # maintaining range window
            self.block_range_window.append(ref_block)
            if len(self.block_range_window) > self.BLOCK_RANGE:
                self.block_range_window.popleft()

            #training
            if len(training_data) > train_amount:
                # logReg_training(model, training_data, label_data, 1, .001)
                model.fit(training_data, label_data)
                del training_data
                del label_data
                training_data = []
                label_data = []
                trained = True
                count_nonevict = 0
                count_evict = 0
            
            # Beginning of standard LIRS logic
            if ref_block == last_ref_block:
                self.pg_hits += 1
                continue
            else:
                pass

            # Is in stack miss ??
            if not self.pg_table[ref_block].is_resident and self.pg_table[ref_block].in_stack:
                self.in_stack_miss += 1
            # Is out stack hit ??
            if self.pg_table[ref_block].is_resident and not self.pg_table[ref_block].in_stack:
                self.out_stack_hit += 1
            if not self.pg_table[ref_block].is_resident and not self.pg_table[ref_block].in_stack:
                self.out_stack_miss += 1

            if not self.pg_table[ref_block].is_resident:
                self.pg_faults += 1
                if self.free_mem == 0:
                    evicted_hir = self.hir_stack.popitem(last=False)
                    self.pg_table[evicted_hir[1]].is_resident = False

                    # gathering training and label data of evicted objects
                    if count_evict <= train_amount/2:
                        training_data.append([self.pg_table[ref_block].reuse_distance, self.pg_table[ref_block].min_b_range,
                                            self.pg_table[ref_block].max_b_range, self.pg_table[ref_block].median])
                        label_data.append(-1)
                        count_evict += 1

                    self.free_mem += 1
                elif self.free_mem > self.HIR_SIZE: # Exception for when the cache is initially empty and is being filled.
                    self.pg_table[ref_block].is_hir = False
                    self.lir_size += 1
                self.free_mem -= 1
            elif self.pg_table[ref_block].is_hir:
                if self.hir_stack.get(ref_block):
                    del self.hir_stack[ref_block]

            if self.pg_table[ref_block].is_resident:
                self.pg_hits += 1

            if self.lir_stack.get(ref_block):
                counter = 0
                for j in self.lir_stack.keys():  # Getting the reuse distance
                    counter += 1
                    if self.lir_stack[j] == ref_block:
                        break
                self.pg_table[ref_block].reuse_distance = (len(self.lir_stack) - counter)  # Setting the reuse distance for that block

                # gathering training data and label data of objects in cache
                if count_nonevict <= train_amount/2:
                    training_data.append([self.pg_table[ref_block].reuse_distance, self.pg_table[ref_block].min_b_range,
                                            self.pg_table[ref_block].max_b_range, self.pg_table[ref_block].median])
                    label_data.append(1)
                    count_nonevict += 1

                del self.lir_stack[ref_block] 
                self.find_lru(self.lir_stack, self.pg_table)

            self.pg_table[ref_block].is_resident = True # Sets the blocks residency in the LIR stack to True.
            self.lir_stack[ref_block] = self.pg_table[ref_block].b_num # Adds the block to the LIR stack.

            if self.pg_table[ref_block].is_hir and self.pg_table[ref_block].in_stack: # Logic for if the block is HIR and is a resident in LIR.
                # If above is true, it makes the HIR block a LIR block.
                self.pg_table[ref_block].is_hir = False
                self.lir_size += 1

                if self.lir_size > mem - self.HIR_SIZE: # If LIR stack is now greater than it's max size, it prunes itself.
                    self.replace_lir_block(self.pg_table, self.lir_size)

            elif self.pg_table[ref_block].is_hir: # If it is an HIR block but not in the stack.
                if not trained: # Standard LIRS logic.
                    self.hir_stack[ref_block] = self.pg_table[ref_block].b_num
                else: # If trained, it now takes advantage of ML model.
                    # prediction = model(torch.sigmoid(torch.tensor([pg_table[ref_block][4], pg_table[ref_block][5],
                    #                                                pg_table[ref_block][6], pg_table[ref_block][7]])
                    #                                  .type(torch.FloatTensor)))

                    feature = np.array([[self.pg_table[ref_block].reuse_distance], [self.pg_table[ref_block].min_b_range],
                                        [self.pg_table[ref_block].max_b_range], [self.pg_table[ref_block].median]])
                    prediction = model.predict(feature.reshape(1, -1))
                    if prediction == 1:
                        self.pg_table[ref_block].is_hir = False
                        self.lir_size += 1
                        if self.lir_size > mem - self.HIR_SIZE:
                            self.replace_lir_block(self.pg_table, self.lir_size)
                    else:
                        self.hir_stack[ref_block] = self.pg_table[ref_block].b_num

            self.pg_table[ref_block].in_stack = True

            last_ref_block = ref_block
        self.result.write(str(self.mem) + ',' + str(self.pg_faults / (self.pg_hits + self.pg_faults) * 100) + ',' + str(self.pg_hits / (self.pg_hits + self.pg_faults) * 100) + "\n")
        self.print_information(mem, self.pg_hits, self.pg_faults, self.HIR_SIZE)
        self.info.write(f"{str(self.mem)},{str(self.in_stack_miss/len(self.trace))},{str(self.out_stack_hit/len(self.trace))}\n")
    
if __name__ == "__main__":
    if (len(sys.argv) != 2):
        raise("Argument Error")

    # Read the trace
    tName = sys.argv[1]
    trace = []
    with open('trace/' + tName, "r") as f:
        for line in f.readlines():
            if not line == "*\n":
                trace.append(int(line))

    # Get the block range of the trace
    vm_size = max(trace)

    # define the name of the directory to be created
    path = "result_set/" + tName
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

    # Store the hit&miss ratio
    result = open("result_set/" + tName + "/ml_lirs_" + tName, "w")

    # in stack miss, out stack hit
    info = open("result_set/" + tName + "/ml_lirs_info_" + tName, "w")

    # Get the trace parameter
    MAX_MEMORY = []
    with codecs.open("cache_size/" + tName, "r", "UTF8") as inputFile:
        inputFile = inputFile.readlines()
    for line in inputFile:
        if not line == "*\n":
            MAX_MEMORY.append(int(line))

    for mem in MAX_MEMORY:
        lirs = LIRS(trace, mem, result, info)
        lirs.ML_LIRS_Replace_Algorithm()
    result.close()

#     f = open("evictions.txt", "w")
#     for i in range(len(eviction_list)):
#         f.write(str(eviction_list[i]) + "\n")
#     f.close()