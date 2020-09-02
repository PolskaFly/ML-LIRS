from collections import deque
from collections import OrderedDict
import codecs
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

def find_lru(s: OrderedDict, pg_tbl: deque):
    temp_s = list(s)
    while pg_tbl[temp_s[0]][2]:
        pg_tbl[temp_s[0]][3] = False
        s.popitem(last=False)
        del temp_s[0]

def get_range(trace) -> int:
    max = 0

    for x in trace:
        if x > max:
            max = x
    return max

def train_model(features, labels, size, rate):
    # Model Training
    features = torch.from_numpy(np.array(features)).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(labels)).type(torch.FloatTensor)

    train_ds = TensorDataset(features, labels)

    batch_size = size
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    next(iter(train_dl))

    model = LogisticRegression(1, 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)

    # m = nn.Sigmoid()
    # training_features = m(training_features)

    for epoch in range(len(features)):
        for i in enumerate(train_dl):
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(features)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()
    return model


def LIRS(pg_hits, pg_faults, free_mem, lir_size, trained):
    training_flt_cntr = 0
    training_hit_cntr = 0
    last_ref_block = -1
    non_zero = False
    inf_count = 0
    for i in range(len(trace)):
        if non_zero:
            if training_hit_cntr + training_flt_cntr > TRAINING_DATA_MIN and not trained:
                m = train_model(training_data, label_data, 30, .01)
                training_data.clear()
                label_data.clear()
                training_flt_cntr = 0
                training_hit_cntr = 0
                trained = True
            elif training_hit_cntr + training_flt_cntr > 30 and trained:
                m = train_model(training_data, label_data, 10, .1)
                training_data.clear()
                label_data.clear()
                training_flt_cntr = 0
                training_hit_cntr = 0

        ref_block = trace[i]

        if ref_block == last_ref_block:
            pg_hits += 1
            continue
        else:
            pass

        if not pg_table[ref_block][1]:
            pg_faults += 1
            if free_mem == 0:
                temp_hir = list(hir_stack)
                pg_table[temp_hir[0]][1] = False

                # Gathering and labeling training data (always happening)
                if i > DATA_COLLECTION_START:
                    if pg_table[temp_hir[0]][4] == 9999 and inf_count < 50:
                        training_data.append([pg_table[temp_hir[0]][4]])
                        label_data.append([0])
                        training_flt_cntr += 1
                        non_zero = True
                        inf_count += 1
                    elif pg_table[temp_hir[0]][4] != 9999 and inf_count >= 50:
                        training_data.append([pg_table[temp_hir[0]][4]])
                        label_data.append([0])
                        training_flt_cntr += 1
                        non_zero = True

                eviction_list.append(temp_hir[0])
                hir_stack.popitem(last=False)
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
            if i > DATA_COLLECTION_START:
                if pg_table[temp_hir[0]][4] == 9999 and inf_count < 50:
                    training_data.append([pg_table[ref_block][4]])
                    label_data.append([1])
                    training_hit_cntr += 1
                    non_zero = True
                    inf_count += 1
                elif pg_table[temp_hir[0]][4] != 9999 and inf_count >= 50:
                    training_data.append([pg_table[ref_block][4]])
                    label_data.append([1])
                    training_hit_cntr += 1
                    non_zero = True

        if lir_stack.get(ref_block):
            temp_lir = list(lir_stack)
            trigger = False
            counter = 0
            for x in range(len(temp_lir)):
                if temp_lir[x] == ref_block:
                    trigger = True
                if trigger:
                    counter += 1
            pg_table[ref_block][4] = counter
            del lir_stack[ref_block]
            find_lru(lir_stack, pg_table)

        pg_table[ref_block][1] = True
        lir_stack[ref_block] = pg_table[ref_block][0]

        if not trained:
            if pg_table[ref_block][2] and pg_table[ref_block][3]:
                pg_table[ref_block][2] = False
                lir_size += 1
                if lir_size > MAX_MEMORY - HIR_SIZE:
                    temp_block = list(lir_stack)[0]
                    pg_table[temp_block][2] = True
                    pg_table[temp_block][3] = False
                    hir_stack[temp_block] = lir_stack[temp_block]
                    find_lru(lir_stack, pg_table)
                    lir_size -= 1
            elif pg_table[ref_block][2]:
                hir_stack[ref_block] = pg_table[ref_block][0]
        else:
            #BIG LOGIC ERROR HERE.
            if m(torch.tensor([pg_table[ref_block][4]]).type(torch.FloatTensor)) > .5 and pg_table[ref_block][3]:
                pg_table[ref_block][2] = False
                lir_size += 1
                if lir_size > MAX_MEMORY - HIR_SIZE:
                    temp_block = list(lir_stack)[0]
                    pg_table[temp_block][2] = True
                    pg_table[temp_block][3] = False
                    hir_stack[temp_block] = lir_stack[temp_block]
                    find_lru(lir_stack, pg_table)
                    lir_size -= 1
            else:
                hir_stack[ref_block] = pg_table[ref_block][0]

        pg_table[ref_block][3] = True

        last_ref_block = ref_block

# Read File In
trace = []
with codecs.open("/Users/polskafly/Desktop/REU/LIRS/traces/2_pools.trc", "r", "UTF8") as inputFile:
    inputFile = inputFile.readlines()
for line in inputFile:
    if not line == "*\n":
        trace.append(int(line))

# Init Parameters
MAX_MEMORY = 100
HIR_PERCENTAGE = 1.0
MIN_HIR_MEMORY = 2

HIR_SIZE = MAX_MEMORY * (HIR_PERCENTAGE / 100)
if HIR_SIZE < MIN_HIR_MEMORY:
    HIR_SIZE = MIN_HIR_MEMORY

TRAINING_HIT_PERCENTAGE = 30.0
TRAINING_DATA_MIN = 500
DATA_COLLECTION_START = MAX_MEMORY * 10
# Init End

#Creating stacks and lists
lir_stack = OrderedDict()
hir_stack = OrderedDict()
pg_table = deque()
eviction_list = []
training_data = deque()
label_data = deque()

# [Block Number, is_resident, is_hir_block, in_stack, reuse distance]
vm_size = get_range(trace)
for x in range(vm_size + 1):
    pg_table.append([x, False, True, False, 9999])

#Creating variables
PG_HITS = 0
PG_FAULTS = 0
free_mem = MAX_MEMORY
lir_size = 0
in_trace = 0
is_trained = False

LIRS(PG_HITS, PG_FAULTS, free_mem, lir_size, is_trained)


print("Hits: ", PG_HITS)
print("Faults: ", PG_FAULTS)
print("Total: ", PG_FAULTS + PG_HITS)
print("HIR Size: ", HIR_SIZE)
print("Hit Ratio: ", PG_HITS/(PG_HITS + PG_FAULTS) * 100)

f = open("evictions.txt", "w")
for i in range(len(eviction_list)):
    f.write(str(eviction_list[i]) + "\n")
f.close()


