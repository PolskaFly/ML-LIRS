from collections import deque
from collections import OrderedDict
import codecs

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


# Read File In
trace = []
with codecs.open("/Users/polskafly/Desktop/REU/LIRS/traces/sprite.trc", "r", "UTF8") as inputFile:
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
# Init End

lir_stack = OrderedDict()
hir_stack = OrderedDict()
pg_table = deque()
eviction_list = []

# [Block Number, is_resident, is_hir_block, in_stack]
vm_size = get_range(trace)
for x in range(vm_size + 1):
    pg_table.append([x, False, True, False])

PG_HITS = 0
PG_FAULTS = 0
free_mem = MAX_MEMORY
lir_size = 0
in_trace = 0
last_ref_block = -1
for i in range(len(trace)):
    ref_block = trace[i]
    #Ask why he makes a duplicate a special case almost...
    if ref_block == last_ref_block:
        PG_HITS += 1
        continue
    else:
        pass

    in_trace += 1
    if not pg_table[ref_block][1]:
        PG_FAULTS += 1
        if free_mem == 0:
            temp_hir = list(hir_stack)
            pg_table[temp_hir[0]][1] = False
            eviction_list.append(temp_hir[0])
            hir_stack.popitem(last=False)
            free_mem += 1
        elif free_mem > HIR_SIZE:
            pg_table[ref_block][2] = False
            lir_size += 1
        free_mem -= 1
    elif pg_table[ref_block][2]:
        try:
            del hir_stack[ref_block]
        except KeyError:
            pass

    if pg_table[ref_block][1]:
        PG_HITS += 1


    if ref_block in lir_stack:
        temp_lir = list(lir_stack)[0]
        del lir_stack[ref_block]
        find_lru(lir_stack, pg_table)

    pg_table[ref_block][1] = True
    lir_stack[ref_block] = pg_table[ref_block][0]


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

    pg_table[ref_block][3] = True

    last_ref_block = ref_block




print("Hits: ", PG_HITS)
print("Faults: ", PG_FAULTS)
print("Total: ", PG_FAULTS + PG_HITS)
print("HIR Size: ", HIR_SIZE)
print("Hit Ratio: ", PG_HITS/(PG_HITS + PG_FAULTS) * 100)

f = open("evictions.txt", "w")
for i in range(len(eviction_list)):
    f.write(str(eviction_list[i]) + "\n")
f.close()


