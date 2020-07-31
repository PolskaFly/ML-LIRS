import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from cache import cache
from lru_cache import lru_cache
import codecs
import collections
import numpy as np

def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb.float())
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
    print(loss)

trace = []
with codecs.open("/Users/polskafly/Desktop/REU/traces/sprite.trc", "r", "UTF8") as inputFile:
    inputFile = inputFile.readlines()
for line in inputFile:
    trace.append(int(line))

trials = input("Trials: ")
trials = int(trials)

runtime_features = []
runtime_labels = []
hash_features = collections.OrderedDict()
hash_labels = collections.OrderedDict()

amount = 0
access_time = 0
lru = lru_cache()

hits = 0
faults = 0

trained = False
trainAmount = 1000

for i in range(trials):
    capacity = input("Capacity: ")
    capacity = int(capacity)
    current_cache = cache(capacity)

    for j in range(len(trace)):
        if len(hash_features) > trainAmount and len(hash_labels) > trainAmount and not trained:
            trained = True

            for key, value in hash_features.items():
                runtime_features.append(hash_features[key])
            for key, value in hash_labels.items():
                if hash_labels[key][1] == -1:
                    runtime_labels.append([100])
                else:
                    runtime_labels.append([(hash_labels[key][1] - hash_labels[key][0])])

            print(runtime_labels)
            print(runtime_features)

            runtime_features = torch.from_numpy(np.array(runtime_features)).float()
            runtime_labels = torch.from_numpy(np.array(runtime_labels)).float()

            train_ds = TensorDataset(runtime_features, runtime_labels)

            batch_size = 30
            train_dl = DataLoader(train_ds, batch_size, shuffle=True)
            next(iter(train_dl))

            model = nn.Linear(1, 1)

            opt = torch.optim.SGD(model.parameters(), lr=.00001)
            loss_fn = F.mse_loss
            loss = loss_fn(model(runtime_features.float()), runtime_labels.float())

            fit(trainAmount, model, loss_fn, opt)

        elif not trained:
            access_time += 1
            lru.set(trace[j], trace[j], current_cache, amount, access_time, hash_labels, hash_features)

       # if trained:
            # hash_features.clear()
            # hash_labels.clear()
            # if trace[j] in current_cache.cache:
            #     hits += 1
            #     current_cache[trace[j]][1] += 1
            #     current_cache.move_key(trace[j])
            # elif len(current_cache.cache) < current_cache.get_capacity():
            #     faults += 1
            #     current_cache.set_cache(trace[j], trace[j], amount)
            # elif len(current_cache.cache) >= current_cache.get_capacity():
            #     #for key, value in current_cache.cache.items():
            #
            #     preds = model(current_cache.cache.items())
            #     print(preds)
            #     current_cache.set_cache(trace[j], trace[j], amount)
            #     faults += 1


    print("Hits", lru.hits)
    print("Faults", lru.faults)
    print("Hit Ratio", ((lru.hits)/(lru.hits + lru.faults))*100)


