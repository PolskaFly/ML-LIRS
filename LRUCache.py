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

class WriteToFile:
    def __init__(self, tName, fp):
        self.tName = tName
        __location__ = "C:\\Users\\Amadeus\\Documents\\Code\\ML_LIRS\\result_set\\" + tName
        self.FILE = open(__location__ + "\\lru_" + fp, "w+")

    def write_to_file(self, *args):
        data = ",".join(args)
        # Store the hit&miss ratio
        self.FILE.write(data + "\n")

class LRUCache:
    def __init__(self, trace, capacity):
        self.capacity = capacity
        self.trace = trace
        self.cache = OrderedDict()
        self.pg_table = deque()
        for x in range(vm_size + 1):
            # [Block Number, is_resident, is_hir_block, in_stack]
            block = Block(x, False, True, False)
            self.pg_table.append(block)
        self.pg_faults = 0
        self.pg_hits = 0
        self.temp_hit = 0
        self.temp_fault = 0
        self.virtual_time = []
        self.inter_hit_ratio = []

    def print_information(self):
        print("Memory Size: ", self.capacity)
        print("Hits: ", self.pg_hits)
        print("Faults: ", self.pg_faults)
        print("Total: ", self.pg_hits + self.pg_faults)
        print("Hit Ratio: ", self.pg_hits / (self.pg_hits + self.pg_faults) * 100)
        return self.capacity, self.pg_faults / (self.pg_faults + self.pg_hits) * 100, self.pg_hits / (
                self.pg_faults + self.pg_hits) * 100, self.inter_hit_ratio, self.virtual_time

    def inter_ratios(self, v_time):
        if v_time % 250 == 0 and v_time != 0:
            h = (self.pg_hits - self.temp_hit) / ((self.pg_faults - self.temp_fault) +
                                                   (self.pg_hits - self.temp_hit)) * 100
            self.inter_hit_ratio.append(float(h))
            self.temp_hit = self.pg_hits
            self.temp_fault = self.pg_faults

            self.virtual_time.append(v_time)

    def set(self, key:int, value:int):
        if key in self.cache:
            self.pg_hits += 1
            self.cache.move_to_end(key)
        elif len(self.cache) < self.capacity:
            self.pg_faults += 1
            self.cache[key] = value
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
            self.cache[key] = value
            self.pg_faults += 1

    def LRU(self):
        count = 0
        for x in self.trace:
            self.inter_ratios(count)
            self.set(self.pg_table[x].b_num, self.pg_table[x].b_num)
            count += 1

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        raise("Argument Error")

    # Read the trace
    tName = sys.argv[1]
    trace = []
    with open('trace/' + tName, "r") as f:
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

    # Get the trace parameter
    MAX_MEMORY = []
    with codecs.open("cache_size/" + tName, "r", "UTF8") as inputFile:
        inputFile = inputFile.readlines()
    for line in inputFile:
        if not line == "*\n":
            MAX_MEMORY.append(int(line))
    FILE = WriteToFile(tName, tName)
    for mem in MAX_MEMORY:
        RATIO_FILE = WriteToFile(tName, str(mem) + "_" + tName + "_ratios")
        v_time = 0
        lru = LRUCache(trace, mem)
        lru.LRU()
        mem, miss, hit, inter_hit_ratio, virtual_time = lru.print_information()
        FILE.write_to_file(str(mem), str(miss), str(hit))
        for i in range(len(inter_hit_ratio)):
            RATIO_FILE.write_to_file(str(virtual_time[i]), str(inter_hit_ratio[i]))
