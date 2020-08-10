import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from cache import cache as C
from lru_cache import lru_cache as lru
import codecs
from collections import OrderedDict

#Does each piece of memory in the cache have data about itself? I think so.
#Sliding memory takes in a certain amount of this data and trains the model off of it?

# Figure out how to implement a sliding memory window to gather data and for labeling. Reuse Distance: After access,
# anything that is above it in the stack is the reuse distance. Recency: Current Virtual Time - Accessed Virtual time
# (Only storing 1 value for the last access) Possibly in future add more access for more data? Train a model. Use
# that model for removal.

trace = []
with codecs.open("/Users/polskafly/Desktop/REU/traces/sprite.trc", "r", "UTF8") as inputFile:
    inputFile = inputFile.readlines()
for line in inputFile:
    trace.append(int(line))

trials = int(input("Trials: "))

sliding_memory_window = []
features = OrderedDict()
training_labels = []
training_features =[]

LRU = lru()

#Inputting trace into cache.
for i in range(trials):
    capacity = int(input("Cache Size: "))
    current_cache = C(capacity)
    sliding_memory_size = capacity / 2
    virtual_time = 0
    for j in range(len(trace)):
        virtual_time += 1

        reuse = 0
        recency = 0

        hasChecked = False

        #Getting reuse distance data.
        if trace[j] in current_cache.cache:
            for key, value in current_cache.get_items():
                if trace[j] == key:
                    hasChecked = True
                elif hasChecked:
                    reuse += 1
            recency = virtual_time - current_cache.cache[trace[j]][1]
            features[trace[j]] = [reuse, recency, max(recency, reuse)]
            training_features.append([reuse, recency, max(recency, reuse)])


        LRU.set(trace[j], trace[j], current_cache, virtual_time)

print(training_features)
print(LRU.get_hits())
print(LRU.get_faults())