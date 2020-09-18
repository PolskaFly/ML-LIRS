from collections import deque
from collections import OrderedDict
import codecs

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

def replace_lir_block(pg_table, lir_size):
    temp_block = lir_stack.popitem(last=False)
    pg_table[temp_block[1]][2] = True
    pg_table[temp_block[1]][3] = False
    hir_stack[temp_block[1]] = temp_block[1]
    find_lru(lir_stack, pg_table)
    lir_size -= 1
    return lir_size

def LIRS(trace, pg_table):
    # Creating variables
    pg_hits = 0
    pg_faults = 0
    free_mem = MAX_MEMORY
    lir_size = 0
    last_ref_block = -1

    for i in range(len(trace)):
        ref_block = trace[i]

        if ref_block == last_ref_block:
            pg_hits += 1
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
            del lir_stack[ref_block]
            find_lru(lir_stack, pg_table)

        pg_table[ref_block][1] = True
        lir_stack[ref_block] = pg_table[ref_block][0]

        if pg_table[ref_block][2] and pg_table[ref_block][3]:
            pg_table[ref_block][2] = False
            lir_size += 1

            if lir_size > MAX_MEMORY - HIR_SIZE:
                replace_lir_block(pg_table, lir_size)

        elif pg_table[ref_block][2]:
            hir_stack[ref_block] = pg_table[ref_block][0]

        pg_table[ref_block][3] = True

        last_ref_block = ref_block

    print_information(pg_hits, pg_faults, HIR_SIZE)


# Read File In
trace = []
with codecs.open("/Users/polskafly/Desktop/REU/LIRS/traces/sprite.trc", "r", "UTF8") as inputFile:
    inputFile = inputFile.readlines()
for line in inputFile:
    if not line == "*\n":
        trace.append(int(line))

# Init Parameters
MAX_MEMORY = 500
HIR_PERCENTAGE = 1.0
MIN_HIR_MEMORY = 2

HIR_SIZE = MAX_MEMORY * (HIR_PERCENTAGE / 100)
if HIR_SIZE < MIN_HIR_MEMORY:
    HIR_SIZE = MIN_HIR_MEMORY
# Init End

#Creating stacks and lists
lir_stack = OrderedDict()
hir_stack = OrderedDict()
pg_tbl = deque()
eviction_list = []

# [Block Number, is_resident, is_hir_block, in_stack]
vm_size = get_range(trace)
for x in range(vm_size + 1):
    pg_tbl.append([x, False, True, False])

LIRS(trace, pg_tbl)

f = open("evictions.txt", "w")
for i in range(len(eviction_list)):
    f.write(str(eviction_list[i]) + "\n")
f.close()


