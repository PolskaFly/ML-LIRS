import sys
import os
import codecs
from collections import deque
from collections import OrderedDict
from sklearn.linear_model import SGDClassifier
import numpy as np
import math

class Block:
    def __init__(self, Block_Number, is_resident, is_hir_block, in_stack):
        self.b_num = Block_Number
        self.is_resident = is_resident
        self.is_hir = is_hir_block
        self.in_stack = in_stack
        self.reuse_distance = -sys.maxsize
        self.ref_times = 0
        self.last_status = None  # LIR HIR


HIR_PERCENTAGE = 1.0
MIN_HIR_MEMORY = 2


class LIRS:
    def __init__(self, trace, mem, result, info):
        self.trace = trace
        # Init the queue
        self.pg_table = deque()
        for x in range(vm_size + 1):
            # [Block Number, is_resident, is_hir_block, in_stack]
            block = Block(x, False, True, False)
            self.pg_table.append(block)

        self.HIR_SIZE = mem * (HIR_PERCENTAGE / 100)
        if self.HIR_SIZE < MIN_HIR_MEMORY:
            self.HIR_SIZE = MIN_HIR_MEMORY
        self.THIRD_SIZE = mem * .01
        # Init End

        # Creating stacks and lists
        self.lir_stack = OrderedDict()
        self.hir_stack = OrderedDict()
        self.train_stack = OrderedDict()
        self.eviction_list = []
        self.pg_hits = 0
        self.pg_faults = 0
        self.free_mem = mem
        self.mem = mem
        self.third_mem = self.THIRD_SIZE
        self.mRatio = 0
        self.in_stack_miss = 0
        self.out_stack_hit = 0
        self.out_stack_miss = 0
        self.lir_size = 0

        self.result = result
        self.info = info
        self.train = False
        self.count_exampler = 0

    def find_boundary(self, h: OrderedDict, table):
        while table[next(iter(h))].is_resident and self.third_mem < self.THIRD_SIZE:
            h.popitem(last=False)

    def find_lru(self, s: OrderedDict, h: OrderedDict, table):
        # Always get first item of the temp_s
        while table[next(iter(s))].is_hir:
            temp = next(iter(s))
            table[temp].in_stack = False
            if temp == next(iter(h)):
                h.popitem(last=False)
                self.third_mem += 1
                self.find_boundary(h, table)
            s.popitem(last=False)

    def block_range(self, x):
        return int(math.floor(x / 100.0)) * 100

    def look_down(self, bottomBlock, s: OrderedDict, h: OrderedDict, table):
        S = list(s)
        check = False
        for key in reversed(S):
            if check:
                h[key] = key
                h.move_to_end(key, last=False)
            if not table[key].is_resident and check:
               # self.third_mem -= 1
                break
            if key == bottomBlock and not check:
                check = True

    def get_range(self, trace) -> int:
        max = 0
        for x in trace:
            if x > max:
                max = x
        return max

    def if_third_access(self, ref_block, s: OrderedDict, h: OrderedDict, table):
        if h.get(ref_block):
            if next(iter(h)) == ref_block:
                self.look_down(h.get(ref_block), s, h, table)
            del h[ref_block]

    def print_information(self, mem, hits, faults, size):
        print("Memory Size: ", mem)
        print("Hits: ", hits)
        print("Faults: ", faults)
        print("Total: ", hits + faults)
        print("Hit Ratio: ", hits / (hits + faults) * 100)

    def replace_lir_block(self, s:OrderedDict, q:OrderedDict, h:OrderedDict, table):
        temp_block = s.popitem(last=False)
        table[temp_block[1]].is_hir = True
        table[temp_block[1]].in_stack = False
        q[temp_block[1]] = temp_block[1]

        self.third_mem = self.THIRD_SIZE
        for key in self.train_stack:
            if not self.pg_table[key].is_resident:
                self.third_mem -= 1

        self.find_lru(s, h, table)
        self.lir_size -= 1
        return self.lir_size

    def third_maintenance(self, third_init, evicted_hir, s: OrderedDict, h: OrderedDict, table):
        if not third_init and s.get(evicted_hir[1]):
            for key in s:
                if key == evicted_hir[1]:
                    h[key] = key
            self.third_mem -= 1
            third_init = True

        if third_init:
            if h.get(evicted_hir[1]) and self.third_mem == 0:
                h.popitem(last=False)
                h[evicted_hir[1]] = evicted_hir[1]
                self.find_boundary(h, table)
            elif h.get(evicted_hir[1]):
                h[evicted_hir[1]] = evicted_hir[1]
                self.third_mem -= 1
                self.find_boundary(h, table)

    def LIRS_Replace_Algorithm(self):
        # Creating variables
        last_ref_block = -1
        third_init = False
        model = SGDClassifier(loss="perceptron", learning_rate="optimal", penalty="L1")

        for i in range(len(self.trace)):
            ref_block = self.trace[i]

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
                    self.third_maintenance(third_init, evicted_hir, self.lir_stack, self.train_stack, self.pg_table)
                    third_init = True
                    self.free_mem += 1
                elif self.free_mem > self.HIR_SIZE:
                    self.pg_table[ref_block].is_hir = False
                    self.lir_size += 1
                self.free_mem -= 1
            elif self.pg_table[ref_block].is_hir:
                if self.hir_stack.get(ref_block):
                    del self.hir_stack[ref_block]

            if self.pg_table[ref_block].is_resident:
                self.pg_hits += 1

            if self.lir_stack.get(ref_block):
                # use the real value and last reference features to update the model

                counter = 0
                for j in self.lir_stack.keys():  # Getting the reuse distance
                    counter += 1
                    if self.lir_stack[j] == ref_block:
                        break
                self.pg_table[ref_block].reuse_distance = (
                            len(self.lir_stack) - counter)  # Setting the reuse distance for that block

                self.if_third_access(ref_block, self.lir_stack, self.train_stack, self.pg_table)
                del self.lir_stack[ref_block]
                self.find_lru(self.lir_stack, self.train_stack, self.pg_table)

            # when use the data to predict the model
            if self.count_exampler == self.mem * 2:
                self.train = True

            self.pg_table[ref_block].is_resident = True
            #if self.train_stack.get(ref_block):
                # self.third_mem += 1

            self.lir_stack[ref_block] = self.pg_table[ref_block].b_num
            if third_init:
                self.train_stack[ref_block] = self.pg_table[ref_block].b_num  # Adds to the third stack

            if self.pg_table[ref_block].is_hir and self.pg_table[ref_block].in_stack:
                self.pg_table[ref_block].is_hir = False
                self.lir_size += 1

                if self.lir_size > mem - self.HIR_SIZE:
                    self.replace_lir_block(self.lir_stack, self.hir_stack, self.train_stack, self.pg_table)

            elif self.pg_table[ref_block].is_hir:

                if not self.train:  # Standard LIRS logic.
                    self.hir_stack[ref_block] = self.pg_table[ref_block].b_num
                else:  # If trained, it now takes advantage of ML model.

                    feature = np.array([[self.pg_table[ref_block].reuse_distance,
                                         self.block_range(self.pg_table[ref_block].b_num)]])

                    prediction = model.predict(feature.reshape(1, -1))

                    # print(prediction)
                    if prediction == 1:
                        self.pg_table[ref_block].is_hir = False
                        self.lir_size += 1
                        if self.lir_size > mem - self.HIR_SIZE:
                            self.replace_lir_block(self.lir_stack, self.hir_stack, self.train_stack, self.pg_table)
                    else:
                        self.hir_stack[ref_block] = self.pg_table[ref_block].b_num
            #testing purposes for accurate size
            self.third_mem = self.THIRD_SIZE
            for key in self.train_stack:
                if not self.pg_table[key].is_resident:
                    self.third_mem -= 1

            if third_init: # had - and self.pg_table[ref_block].reuse_distance != -sys.maxsize
                model.partial_fit(np.array([[self.pg_table[ref_block].reuse_distance,
                                             self.block_range(self.pg_table[ref_block].b_num)]]),
                                  np.array([1 if self.train_stack.get(ref_block) else -1]),
                                  classes=np.array([1, -1]))
                self.count_exampler += 1

            self.pg_table[ref_block].in_stack = True
            self.pg_table[ref_block].ref_times += 1
            last_ref_block = ref_block

        self.result.write(
            f"{str(self.mem)},{str(self.pg_faults / (self.pg_hits + self.pg_faults) * 100)},{str(self.pg_hits / (self.pg_hits + self.pg_faults) * 100)}\n")
        self.print_information(mem, self.pg_hits, self.pg_faults, self.HIR_SIZE)
        self.info.write(
            f"{str(self.mem)},{str(self.in_stack_miss / len(self.trace))},{str(self.out_stack_hit / len(self.trace))}\n")
        print("in stack miss: ", self.in_stack_miss, "out stack hit: ", self.out_stack_hit, "out stack miss: ",
              self.out_stack_miss)


if __name__ == "__main__":
    if (len(sys.argv) != 2):
        raise ("Argument Error")

    # Read the trace
    tName = sys.argv[1]
    trace = []
    with open('trace/' + tName, "r") as f:
        for line in f.readlines():
            if not line == "*\n" and not line == "\n":
                trace.append(int(line.rstrip()))

    # Get the block range of the trace
    vm_size = max(trace)

    # define the name of the directory to be created
    path = "result_set/" + tName
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

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
        lirs.LIRS_Replace_Algorithm()
    result.close()