import numpy as np
import matplotlib.pyplot as plt
import codecs
import os
import sys

WORK_SPACE = './myLirs/'

markers = ['X', 'o', 'v', '.', '+', '1', '-', 't']
algo = ['LIRS', 'LRU', 'ML-LIRS', 'OPT', 'LIRS BD Ratio', 'ML-LIRS BD Ratio']
colors = ['r', 'g', 'k', 'y', 'm', 'b', 'b', 'b']

def plot(X, Y, tName):
    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)
    #plt.grid(linestyle='-')
    for i, y in enumerate(Y):
        y = [float(_) for _ in y]
        plt.plot(X, y, color=colors[i], marker=markers[i], label=algo[i], alpha=0.6)
    # plt.title(tName)
    plt.xlabel('Cache Size (# of blocks)', size=11)
    plt.ylabel('Miss Rate(%)', size=11)
    plt.legend()
    """
    Set y axis begin at 0
    """
    plt.ylim(bottom=0)
    plt.savefig("graph/miss_ratio/" + tName, bbox_inches='tight', pad_inches=0)
    plt.close()
        
def get_result(path):
    print(path + "..")
    res = []
    t = []
    with open(path, 'r') as f:
        for flt in f.readlines():
            """
            mCacheMaxLimit,
            missRatio, 
            100 - missRatio);
            """
            flt = flt.strip().split(',')
            if not flt:
                continue
            cache_size = int(flt[0].strip('l'))
            try:
                t.append( (cache_size, flt[1], flt[2], flt[3]))
            except:
                t.append( (cache_size, flt[1], flt[2]))
        t.sort(key=lambda x: x[0])
        
        for tp in t:
            res.append(float(tp[1]))
        
        return res

def get_result_bd(path):
    print(path + "..")
    res = []
    t = []
    with open(path, 'r') as f:
        for flt in f.readlines():
            """
            mCacheMaxLimit,
            missRatio, 
            100 - missRatio);
            """
            flt = flt.strip().split(',')
            if not flt:
                continue
            cache_size = int(flt[0].strip('l'))
            try:
                t.append( (cache_size, flt[1], flt[2], flt[3]))
            except:
                t.append( (cache_size, flt[1], flt[2]))
        t.sort(key=lambda x: x[0])
        
        for tp in t:
            res.append(float(tp[3]))
        
        return res
    
    
if __name__ == "__main__":
    if (len(sys.argv) != 2):
        raise("Argument Error")
    tName = sys.argv[1]

    miss_rate_set = []
    miss_rate_set.append(get_result(os.path.realpath(os.path.abspath("result_set\\" + tName + "\\lirs_" + tName))))
    miss_rate_set.append(get_result(os.path.realpath(os.path.abspath("result_set\\" + tName + "\\lru_" + tName))))
    miss_rate_set.append(get_result(os.path.realpath(os.path.abspath("result_set\\" + tName + "\\ml_lirs_v9_" + tName))))
    miss_rate_set.append(get_result(os.path.realpath(os.path.abspath("result_set\\" + tName + "\\opt_" + tName))))

    miss_rate_set.append(get_result_bd(os.path.realpath(os.path.abspath("result_set\\" + tName + "\\lirs_" + tName))))
    miss_rate_set.append(get_result_bd(os.path.realpath(os.path.abspath("result_set\\" + tName + "\\ml_lirs_v9_" + tName))))

    # Get the trace parameter
    MAX_MEMORY = []
    with codecs.open("cache_size\\" + tName, "r", "UTF8") as inputFile:
        inputFile = inputFile.readlines()
    for line in inputFile:
        if not line == "*\n":
            MAX_MEMORY.append(int(line))
    plot(MAX_MEMORY, miss_rate_set, tName)


