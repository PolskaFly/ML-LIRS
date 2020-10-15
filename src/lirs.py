from collections import deque
from collections import OrderedDict
import codecs
import sys
import os

class Block:
    def __init__(self, Block_Number, is_resident, is_hir_block, in_stack):
        self.b_num = Block_Number
        self.is_resident = is_resident
        self.is_hir = is_hir_block
        self.in_stack = in_stack

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

    def LIRS_Replace_Algorithm(self):
        # Creating variables
        last_ref_block = -1

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
                del self.lir_stack[ref_block]
                self.find_lru(self.lir_stack, self.pg_table)

            self.pg_table[ref_block].is_resident = True
            self.lir_stack[ref_block] = self.pg_table[ref_block].b_num

            if self.pg_table[ref_block].is_hir and self.pg_table[ref_block].in_stack:
                self.pg_table[ref_block].is_hir = False
                self.lir_size += 1

                if self.lir_size > mem - self.HIR_SIZE:
                    self.replace_lir_block(self.pg_table, self.lir_size)

            elif self.pg_table[ref_block].is_hir:
                self.hir_stack[ref_block] = self.pg_table[ref_block].b_num

            self.pg_table[ref_block].in_stack = True

            last_ref_block = ref_block
        self.result.write(f"{str(self.mem)},{str(self.pg_faults / (self.pg_hits + self.pg_faults) * 100)},{str(self.pg_hits / (self.pg_hits + self.pg_faults) * 100)}\n")
        self.print_information(mem, self.pg_hits, self.pg_faults, self.HIR_SIZE)
        self.info.write(f"{str(self.mem)},{str(self.in_stack_miss/len(self.trace))},{str(self.out_stack_hit/len(self.trace))}\n")
        print("in stack miss: ", self.in_stack_miss, "out stack hit: ", self.out_stack_hit, "out stack miss: ", self.out_stack_miss)
    
if __name__ == "__main__":
    if (len(sys.argv) != 2):
        raise("Argument Error")

    # Read the trace
    tName = sys.argv[1]
    trace = []
    with open('../trace/' + tName, "r") as f:
        for line in f.readlines():
            if not line == "*\n":
                trace.append(int(line))

    # Get the block range of the trace
    vm_size = max(trace)

    # define the name of the directory to be created
    path = "../result_set/" + tName
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

    # Store the hit&miss ratio
    result = open("../result_set/" + tName + "/lirs_" + tName, "w")

    # in stack miss, out stack hit
    info = open("../result_set/" + tName + "/lirs_info_" + tName, "w")

    # Get the trace parameter
    MAX_MEMORY = []
    with codecs.open("../cache_size/" + tName, "r", "UTF8") as inputFile:
        inputFile = inputFile.readlines()
    for line in inputFile:
        if not line == "*\n":
            MAX_MEMORY.append(int(line))

    for mem in MAX_MEMORY:
        lirs = LIRS(trace, mem, result, info)
        lirs.LIRS_Replace_Algorithm()
    result.close()
    info.close()

#     f = open("evictions.txt", "w")
#     for i in range(len(eviction_list)):
#         f.write(str(eviction_list[i]) + "\n")
#     f.close()