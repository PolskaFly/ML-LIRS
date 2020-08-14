import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from cache import cache as C
from lru_cache import lru_cache as lru
import codecs
from collections import OrderedDict
import numpy as np
import virtualtime


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x.float())
        return out


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

#Inputting trace into cache.
for i in range(trials):
    capacity = int(input("Cache Size: "))
    current_cache = C(capacity)
    sliding_memory_size = capacity/2
    virtual_time = 0
    trained = False
    LRU = lru()
    hits = 0
    faults = 0

    # current_cache.set_cache(10, 5, 10, 5, False)
    # current_cache.set_cache(6,9, 10, 5, False)
    #
    # print(current_cache.cache)
    # print(current_cache.cache[list(current_cache.cache.keys())[-1]][1])

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

            recency = virtual_time - current_cache.get_access(trace[j])
            training_features.append([reuse, recency, max(recency, reuse)])
        if len(training_features) >= 500:
            for i in range(len(training_features)):
                if training_features[i][1] > sliding_memory_size:
                    training_labels.append([0])
                else:
                    training_labels.append([1])

            # Model Training
            ttraining_features = torch.from_numpy(np.array(training_features)).float()
            ttraining_labels = torch.from_numpy(np.array(training_labels)).float()

            train_ds = TensorDataset(ttraining_features, ttraining_labels)

            batch_size = 30
            train_dl = DataLoader(train_ds, batch_size, shuffle=True)
            next(iter(train_dl))

            model = LogisticRegression(3, 1)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=.001)

            for epoch in range(len(training_features)):
                for i in enumerate(train_dl):
                    # Clear gradients w.r.t. parameters
                    optimizer.zero_grad()

                    # Forward pass to get output/logits
                    outputs = model(ttraining_features.float())

                    # Calculate Loss: softmax --> cross entropy loss
                    loss = criterion(outputs.float(), ttraining_labels.float())

                    # Getting gradients w.r.t. parameters
                    loss.backward()

                    # Updating parameters
                    optimizer.step()

            trained = True

            training_features.clear()
            training_labels.clear()

        if not trained:
            LRU.set(trace[j], trace[j], current_cache, virtual_time)
        else:
            if hasChecked:
                hits += 1
                current_cache.set_cache(trace[j], trace[j], reuse, recency)
            elif len(current_cache.cache) < current_cache.get_capacity():
                faults += 1
                current_cache.set_cache(trace[j], trace[j], 0, virtual_time)
            elif len(current_cache.cache) >= current_cache.get_capacity():
                if model(torch.from_numpy(np.array(current_cache.cache[list(current_cache.cache.keys())[0]][1])).float()) > .5:
                    for key in list(current_cache.cache.keys()):
                        if current_cache.get_access(key) > sliding_memory_size:
                            current_cache.set_cache(key, key, 1 + (sliding_memory_size * 2), sliding_memory_size * 2)
                        else:
                            current_cache.set_cache(key, key, sliding_memory_size * 2, virtual_time - current_cache.get_access(key))

                        if model(torch.from_numpy(np.array(current_cache.cache[key][1])).float()) >= .5:
                            current_cache.cache.move_to_end(key)

                current_cache.pop(list(current_cache.cache.keys())[0])
                current_cache.set_cache(trace[j], trace[j], sliding_memory_size * 2, virtual_time, virtual_time)

                faults += 1

    print(LRU.get_hits())
    print(LRU.get_faults())

    print("Total H", LRU.get_hits() + hits)
    print("Total F", LRU.get_faults() + faults)

print(current_cache.cache)
print("Features", training_features)
