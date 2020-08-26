from collections import deque
from collections import OrderedDict
import codecs

def find_lru(s: OrderedDict, pg_tbl: deque):
    temp_s = list(s.keys())
    while pg_tbl[temp_s[0]][2]:
        pg_tbl[temp_s[0]][3] = False
        s.popitem(last=False)
        del temp_s[0]

def get_range(trace) -> int:
    min = 0
    max = 0

    for x in trace:
        if x > max:
            max = x
        if x <= min:
            min = x

    return max


# Read File In
trace = []
with codecs.open("/Users/polskafly/Desktop/REU/LIRS/traces/2_pools.trc", "r", "UTF8") as inputFile:
    inputFile = inputFile.readlines()
for line in inputFile:
    if not line == "*\n":
        trace.append(int(line))

# Init Parameters
MAX_MEMORY = 400
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

for i in range(len(trace)):
    ref_block = trace[i]

    if not pg_table[ref_block][1]:
        PG_FAULTS += 1
        if free_mem == 0:
            temp_hir = list(hir_stack)
            pg_table[temp_hir[0]][1] = False
            eviction_list.append(hir_stack[temp_hir[0]])
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
    try:
        del lir_stack[ref_block]
    except KeyError:
        pass

    pg_table[ref_block][1] = True
    lir_stack[ref_block] = pg_table[ref_block][0]

    if pg_table[ref_block][2] and pg_table[ref_block][3]:
        pg_table[ref_block][2] = False
        lir_stack[ref_block] = pg_table[ref_block][0]
        lir_size += 1
        if lir_size > MAX_MEMORY - HIR_SIZE:
            temp_block = list(lir_stack.keys())[0]
            pg_table[temp_block][2] = True
            pg_table[temp_block][3] = False
            hir_stack[temp_block] = pg_table[temp_block][0]
            lir_stack[temp_block] = pg_table[temp_block][0]
            find_lru(lir_stack, pg_table)
            lir_size -= 1
    elif pg_table[ref_block][2]:
        hir_stack[ref_block] = pg_table[ref_block][0]

    pg_table[ref_block][3] = True



print(PG_HITS)
print(PG_FAULTS)
print(PG_FAULTS + PG_HITS)

f = open("evictions.txt", "w")
for i in range(len(eviction_list)):
    f.write(str(eviction_list[i]) + "\n")
f.close()


