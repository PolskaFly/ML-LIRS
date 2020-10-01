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


def find_lru(s: OrderedDict, pg_table: deque):
    temp_s = list(s)
    while pg_table[temp_s[0]][2]:
        pg_table[temp_s[0]][3] = False
        s.popitem(last=False)
        del temp_s[0]

def get_range(trace) -> int:
    max = 0
    for x in trace:
        if x > max:
            max = x
    return max

def print_information(hits, faults, size):
    print("Hits: ", hits)
    print("Faults: ", faults)
    print("Total: ", hits + faults)
    print("HIR Size: ", size)
    print("Hit Ratio: ", hits / (hits + faults) * 100)

def logReg_training(model, features, labels, size, rate):
    temp_features = torch.from_numpy(np.array(features)).type(torch.FloatTensor)
    temp_labels = torch.from_numpy(np.array(labels)).type(torch.FloatTensor)

    train_ds = TensorDataset(temp_features, temp_labels)

    batch_size = size
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    next(iter(train_dl))

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)

    # Clear gradients w.r.t. parameters
    optimizer.zero_grad()

    # Forward pass to get output/logits
    outputs = model(temp_features)

    outputs = torch.sigmoid(outputs)

    # Calculate Loss
    loss = criterion(outputs, temp_labels)

    # Getting gradients w.r.t. parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    return model

def replace_lir_block(pg_table, lir_size):
    temp_block = lir_stack.popitem(last=False)
    pg_table[temp_block[1]][2] = True
    pg_table[temp_block[1]][3] = False
    hir_stack[temp_block[1]] = temp_block[1]
    find_lru(lir_stack, pg_table)
    lir_size -= 1
    return lir_size

def LIRS(trace, pg_table, block_range_window):
    # Creating variables
    pg_hits = 0
    pg_faults = 0
    free_mem = MAX_MEMORY
    lir_size = 0
    last_ref_block = -1
    trained = False
    training_data = deque()
    label_data = deque()
    train_amount = 100

    count_evict = 0
    count_nonevict = 0


    #initializing model
    # model = LogisticRegression(4, 1)
    model = GradientBoostingClassifier(n_estimators=30, learning_rate=.1, max_features=2, max_depth=15,
                                        random_state=0)

    for i in range(len(trace)):
        ref_block = trace[i]

        # setting block range
        if len(block_range_window) == BLOCK_RANGE:
            pg_table[ref_block][5] = min(block_range_window)
            pg_table[ref_block][6] = max(block_range_window)
            pg_table[ref_block][7] = median(block_range_window)

        # maintaining range window
        block_range_window.append(ref_block)
        if len(block_range_window) > BLOCK_RANGE:
            block_range_window.popleft()

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

        #Beginning of standard LIRS logic
        if ref_block == last_ref_block:
            pg_hits += 1
            pg_table[ref_block][4] = 0
            continue
        else:
            pass

        if not pg_table[ref_block][1]:
            pg_faults += 1
            if free_mem == 0:
                evicted_hir = hir_stack.popitem(last=False)
                pg_table[evicted_hir[1]][1] = False

                # gathering training and label data of evicted objects
                if count_evict <= train_amount/2:
                    training_data.append([pg_table[ref_block][4], pg_table[ref_block][5],
                                          pg_table[ref_block][6], pg_table[ref_block][7]])
                    label_data.append(-1)
                    count_evict += 1

                free_mem += 1
            elif free_mem > HIR_SIZE:  # Exception for when the cache is initially empty and is being filled.
                pg_table[ref_block][2] = False
                lir_size += 1
            free_mem -= 1

        elif pg_table[ref_block][2]:
            if hir_stack.get(ref_block):
                del hir_stack[ref_block]

        if pg_table[ref_block][1]:
            pg_hits += 1

        if lir_stack.get(ref_block):
            counter = 0
            for j in lir_stack.keys():  # Getting the reuse distance
                counter += 1
                if lir_stack[j] == ref_block:
                    break
            pg_table[ref_block][4] = (len(lir_stack) - counter)  # Setting the reuse distance for that block

            # gathering training data and label data of objects in cache
            if count_nonevict <= train_amount/2:
                training_data.append([pg_table[ref_block][4], pg_table[ref_block][5],
                                        pg_table[ref_block][6], pg_table[ref_block][7]])
                label_data.append(1)
                count_nonevict += 1


            del lir_stack[ref_block]
            find_lru(lir_stack, pg_table)

        pg_table[ref_block][1] = True  # Sets the blocks residency in the LIR stack to True.
        lir_stack[ref_block] = pg_table[ref_block][0]  # Adds the block to the LIR stack.

        if pg_table[ref_block][2] and pg_table[ref_block][3]:  # Logic for if the block is HIR and is a resident in LIR.
            # If above is true, it makes the HIR block a LIR block.
            pg_table[ref_block][2] = False
            lir_size += 1

            if lir_size > MAX_MEMORY - HIR_SIZE:  # If LIR stack is now greater than it's max size, it prunes itself.
                replace_lir_block(pg_table, lir_size)

        elif pg_table[ref_block][2]:  # If it is an HIR block but not in the stack.
            if not trained:  # Standard LIRS logic.
                hir_stack[ref_block] = pg_table[ref_block][0]
            else:  # If trained, it now takes advantage of ML model.
                # prediction = model(torch.sigmoid(torch.tensor([pg_table[ref_block][4], pg_table[ref_block][5],
                #                                                pg_table[ref_block][6], pg_table[ref_block][7]])
                #                                  .type(torch.FloatTensor)))

                feature = np.array([[pg_table[ref_block][4]], [pg_table[ref_block][5]],
                                    [pg_table[ref_block][6]], [pg_table[ref_block][7]]])
                prediction = model.predict(feature.reshape(1, -1))
                if prediction == 1:
                    pg_table[ref_block][2] = False
                    lir_size += 1
                    if lir_size > MAX_MEMORY - HIR_SIZE:
                        replace_lir_block(pg_table, lir_size)
                else:
                    hir_stack[ref_block] = pg_table[ref_block][0]

        pg_table[ref_block][3] = True  # Sets it's recency in the LIR stack to true.

        last_ref_block = ref_block


    print_information(pg_hits, pg_faults, HIR_SIZE)


# Read File In
trace_array = []
with codecs.open("/Users/polskafly/Desktop/REU/LIRS/traces/ps.trc", "r", "UTF8") as inputFile:
    inputFile = inputFile.readlines()
for line in inputFile:
    if not line == "*\n" or not line == "\n" or not line == '*\n':
        trace_array.append(int(line))

# Init Parameters
MAX_MEMORY = 700
HIR_PERCENTAGE = 1.0
MIN_HIR_MEMORY = 2

HIR_SIZE = MAX_MEMORY * (HIR_PERCENTAGE / 100)
if HIR_SIZE < MIN_HIR_MEMORY:
    HIR_SIZE = MIN_HIR_MEMORY

BLOCK_RANGE = HIR_SIZE * 5

if BLOCK_RANGE > 15:
    BLOCK_RANGE = 15
# Init End

#Creating stacks and lists
lir_stack = OrderedDict()
hir_stack = OrderedDict()
pg_tbl = deque()
eviction_list = []
block_range_window = deque()

# [Block Number, is_resident, is_hir_block, in_stack, reuse distance, min block range, max block range, median]
vm_size = get_range(trace_array)
for x in range(vm_size + 1):
    pg_tbl.append([x, False, True, False, -1, 0, 0, 0])


LIRS(trace_array, pg_tbl, block_range_window)

f = open("evictions.txt", "w")
for i in range(len(eviction_list)):
    f.write(str(eviction_list[i]) + "\n")
f.close()


