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
        self.prediction = -1

HIR_PERCENTAGE = 1.0
MIN_HIR_MEMORY = 2

class WriteToFile:
    def __init__(self, tName, fp):
        self.tName = tName
        __location__ = "C:\\Users\\Amadeus\\Documents\\Code\\ML_LIRS\\result_set\\" + tName
        self.FILE = open(__location__ + "\\lirs_" + fp, "w+")

    def write_to_file(self, *args):
        data = ",".join(args)
        # Store the hit&miss ratio
        self.FILE.write(data + "\n")

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
        self.temp_hit = 0
        self.temp_fault = 0

        self.virtual_time = []
        self.inter_hit_ratio = []

        self.good_pred = 0
        self.bad_pred = 0

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

    def print_information(self):
        print("Memory Size: ", self.mem)
        print("Hits: ", self.pg_hits)
        print("Faults: ", self.pg_faults)
        print("Total: ", self.pg_hits + self.pg_faults)
        print("Hit Ratio: ", self.pg_hits / (self.pg_hits + self.pg_faults) * 100)
        print("Good Decisions:", self.good_pred)
        print("Bad Decisions:", self.bad_pred)
        print("Bad Decision Ratio: ", (self.bad_pred/(self.bad_pred+self.good_pred)) * 100)
        return self.mem, self.pg_faults / (self.pg_faults + self.pg_hits) * 100, self.pg_hits / (
                self.pg_faults + self.pg_hits) * 100, self.inter_hit_ratio, self.virtual_time, (self.bad_pred/(self.bad_pred+self.good_pred)) * 100

    def inter_ratios(self, v_time):
        if v_time % 250 == 0 and v_time != 0:
            h = (self.pg_hits - self.temp_hit) / ((self.pg_faults - self.temp_fault) +
                                                   (self.pg_hits - self.temp_hit)) * 100
            self.inter_hit_ratio.append(float(h))
            self.temp_hit = self.pg_hits
            self.temp_fault = self.pg_faults

            self.virtual_time.append(v_time)

    def replace_lir_block(self, pg_table, lir_size):
        temp_block = self.lir_stack.popitem(last=False)
        self.pg_table[temp_block[1]].is_hir = True
        self.pg_table[temp_block[1]].in_stack = False
        self.hir_stack[temp_block[1]] = temp_block[1]
        self.find_lru(self.lir_stack, pg_table)
        self.lir_size -= 1
        return self.lir_size

    def LIRS_Replace_Algorithm(self, v_time):
        # Creating variables
        last_ref_block = -1

        for i in range(len(self.trace)):
            ref_block = self.trace[i]

            self.inter_ratios(i)

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

            if self.pg_table[ref_block].prediction == 1 and not self.pg_table[ref_block].is_resident and not self.pg_table[ref_block].in_stack:
                self.good_pred += 1
            elif self.pg_table[ref_block].prediction == 1 and (self.pg_table[ref_block].is_resident or self.pg_table[ref_block].in_stack):
                self.bad_pred += 1
            elif self.pg_table[ref_block].prediction == 0 and self.pg_table[ref_block].is_resident:
                self.good_pred += 1
            elif self.pg_table[ref_block].prediction == 0 and not self.pg_table[ref_block].is_resident:
                self.bad_pred += 1

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

            if self.pg_table[ref_block].is_hir:
                self.pg_table[ref_block].prediction = 1
            else:
                self.pg_table[ref_block].prediction = 0

            self.pg_table[ref_block].in_stack = True

            last_ref_block = ref_block
    
if __name__ == "__main__":
    if (len(sys.argv) != 2):
        raise("Argument Error")

    # Read the trace
    tName = sys.argv[1]
    trace = []
    with open('trace\\' + tName, "r") as f:
        for line in f.readlines():
            if not line == "*\n" and not line == "\n":
                trace.append(int(line))

    # Get the block range of the trace
    vm_size = max(trace)

    # define the name of the directory to be created
    path = "result_set\\" + tName
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

    # Store the hit&miss ratio
    result = 0
    info = 0
    # in stack miss, out stack hit
    # info = open("result_set/" + tName + "/lirs_info_" + tName, "w")

    # Get the trace parameter
    MAX_MEMORY = []
    with codecs.open("cache_size\\" + tName, "r", "UTF8") as inputFile:
        inputFile = inputFile.readlines()
    for line in inputFile:
        if not line == "*\n":
            MAX_MEMORY.append(int(line))
    FILE = WriteToFile(tName, tName)
    for mem in MAX_MEMORY:
        RATIO_FILE = WriteToFile(tName, str(mem) + "_" + tName + "_ratios")
        v_time = 0
        lirs = LIRS(trace, mem, result, info)
        lirs.LIRS_Replace_Algorithm(v_time)
        mem, miss, hit, inter_hit_ratio, virtual_time, bad_decision_ratio= lirs.print_information()
        FILE.write_to_file(str(mem), str(miss), str(hit), str(bad_decision_ratio))
        for i in range(len(inter_hit_ratio)):
            RATIO_FILE.write_to_file(str(virtual_time[i]), str(inter_hit_ratio[i]))
