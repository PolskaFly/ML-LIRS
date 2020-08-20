from collections import deque
from collections import OrderedDict
import codecs

#Two Stacks: S and Q
def find_LIR(S: OrderedDict, pg: deque):
    while S[-1][2]:
        pg[S[-1]][3] = False
        S.popitem(last=False)
    return S

def set_hir_block(pg_table, ref_block, bool):
    pg_table[ref_block][2] = bool

def set_residence(pg_table, ref_block, bool):
    pg_table[ref_block][1] = bool

def set_recency(pg_table, ref_block, bool):
    pg_table[ref_block][3] = bool

def get_range(trace) -> int:
    min = 0
    max = 0

    for x in trace:
        if x > max:
            max = x
        if x <= min:
            min = x

    return max

#Read File In
trace = []
with codecs.open("/Users/polskafly/Desktop/REU/traces/sprite.trc", "r", "UTF8") as inputFile:
    inputFile = inputFile.readlines()
for line in inputFile:
    trace.append(int(line))

#Initialization Parameters
MAX_MEMORY = 100
HIR_PERCENTAGE = 1.0

HIR_SIZE = MAX_MEMORY * HIR_PERCENTAGE/100
#Init End

lir_stack = OrderedDict()
hir_stack = OrderedDict()
pg_table = deque()

#[Block Number, is_resident, is_hir_block, in_stack]
vm_size = get_range(trace)
for x in range(vm_size):
    pg_table.append([x, False, True, False])

PG_HITS = 0
PG_FAULTS = 0
free_mem = MAX_MEMORY

for i in range(len(trace)):
    ref_block = trace[i]
    if not pg_table[i][1]:
        PG_FAULTS += 1

        if free_mem == 0:
            #pg_table[hir_stack[-1]][1] = False
            set_residence(pg_table, list(hir_stack.items())[-1][1][0], False)
            hir_stack.popitem(last=False)
            free_mem += 1
        elif free_mem > HIR_SIZE:
            set_hir_block(pg_table, ref_block, False)
        free_mem -= 1
    elif pg_table[ref_block][2] and not pg_table[ref_block][3]:
        hir_stack.pop(ref_block)
        hir_stack.move_to_end(ref_block)
        PG_HITS += 1
    else:
        PG_HITS += 1
        lir_stack.move_to_end(ref_block)

    lir_stack[ref_block] = pg_table[ref_block]
    lir_stack.move_to_end(ref_block)
    set_residence(pg_table, ref_block, True)

    if not pg_table[ref_block][3]:
        a = 1

    if pg_table[ref_block][2] and pg_table[ref_block][3]:
        set_hir_block(pg_table, ref_block, False)

    if len(lir_stack) > MAX_MEMORY - HIR_SIZE:
        set_hir_block(pg_table, ref_block, True)
        set_recency(pg_table, ref_block, False)
        hir_stack[ref_block] = pg_table[ref_block]
        find_LIR(lir_stack, pg_table)
    elif pg_table[ref_block][3]:
        hir_stack[ref_block] = pg_table[ref_block]

    pg_table[ref_block][3] = True

print(PG_HITS)
print(PG_FAULTS)