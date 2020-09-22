from collections import deque
from collections import OrderedDict
import codecs
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from statistics import median

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

def online_training(model, features, labels, size, rate):
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

    #initializing model
    model = LogisticRegression(3, 1)

    for i in range(len(trace)):
        ref_block = trace[i]

        if len(training_data) > 20:
            online_training(model, training_data, label_data, 5, .001)
            training_data.clear()
            label_data.clear()
            trained = True

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
                free_mem += 1
            elif free_mem > HIR_SIZE:
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
            for j in lir_stack.keys():  # Reuse distance
                counter += 1
                if lir_stack[j] == ref_block:
                    break
            pg_table[ref_block][4] = (len(lir_stack) - counter)
            del lir_stack[ref_block]
            find_lru(lir_stack, pg_table)

        if len(block_range_window) == BLOCK_RANGE:
            pg_table[ref_block][5] = min(block_range_window)
            pg_table[ref_block][6] = max(block_range_window)
            pg_table[ref_block][7] = median(block_range_window)

        pg_table[ref_block][1] = True
        lir_stack[ref_block] = pg_table[ref_block][0]

        if pg_table[ref_block][2] and pg_table[ref_block][3]:
            pg_table[ref_block][2] = False
            lir_size += 1

            if lir_size > MAX_MEMORY - HIR_SIZE:
                replace_lir_block(pg_table, lir_size)

        elif pg_table[ref_block][2]:
            if not trained or pg_table[ref_block][4] == 9999:
                hir_stack[ref_block] = pg_table[ref_block][0]
            else:
                prediction = model(torch.sigmoid(torch.tensor([pg_table[ref_block][5],
                                                               pg_table[ref_block][6], pg_table[ref_block][6]])
                                                 .type(torch.FloatTensor)))
                if prediction > .5:
                    pg_table[ref_block][2] = False
                    lir_size += 1
                    if lir_size > MAX_MEMORY - HIR_SIZE:
                        replace_lir_block(pg_table, lir_size)
                else:
                    hir_stack[ref_block] = pg_table[ref_block][0]

        pg_table[ref_block][3] = True

        block_range_window.append(ref_block)
        if len(block_range_window) > BLOCK_RANGE:
            block_range_window.popleft()  # check if this does what I want

        last_ref_block = ref_block

        if pg_table[ref_block][4] != 9999:
            training_data.append([pg_table[ref_block][5],
                                  pg_table[ref_block][6], pg_table[ref_block][7]])
            if pg_table[ref_block][1] and not pg_table[ref_block][2]:
                label_data.append([1])
            else:
                label_data.append([0])

    print_information(pg_hits, pg_faults, HIR_SIZE)


# Read File In
trace_array = []
with codecs.open("/Users/polskafly/Desktop/REU/LIRS/traces/multi1.trc", "r", "UTF8") as inputFile:
    inputFile = inputFile.readlines()
for line in inputFile:
    if not line == "*\n" or not line == "\n":
        trace_array.append(int(line))

# Init Parameters
MAX_MEMORY = 400
HIR_PERCENTAGE = 1.0
MIN_HIR_MEMORY = 2
BLOCK_RANGE = 4

HIR_SIZE = MAX_MEMORY * (HIR_PERCENTAGE / 100)
if HIR_SIZE < MIN_HIR_MEMORY:
    HIR_SIZE = MIN_HIR_MEMORY
# Init End

#Creating stacks and lists
lir_stack = OrderedDict()
hir_stack = OrderedDict()
pg_tbl = deque()
eviction_list = []
training_data = deque()
label_data = deque()
block_range_window = deque()

# [Block Number, is_resident, is_hir_block, in_stack, reuse distance, min block range, max block range, median]
vm_size = get_range(trace_array)
for x in range(vm_size + 1):
    pg_tbl.append([x, False, True, False, 9999, 0, 0, 0])


LIRS(trace_array, pg_tbl, block_range_window)

f = open("evictions.txt", "w")
for i in range(len(eviction_list)):
    f.write(str(eviction_list[i]) + "\n")
f.close()


