import numpy as np
import matplotlib.pyplot as plt
import codecs
import sys
import os

WORK_SPACE = './myLirs/'

markers = ['X', 'o', 'v', '.', '+', '1']
algo = ['LIRS', 'LRU', 'ML-LIRS', 'OPT']
colors = ['r', 'g', 'k', 'y', 'm', 'b']


def plot(X, Y1, Y2, tName, size):
    x = [int(_) for _ in X]
    y1 = [float(_) for _ in Y1]
    plt.plot(x, y1, color=colors[0], marker=markers[0], label= ('hit_ratio_' + size), alpha=0.6)
    y2 = [float(_) for _ in Y2]
    plt.plot(x, y2, color=colors[1], marker=markers[1], label= ('d_ratio_' + size + "(recency0/total)"), alpha=0.6)
    plt.title(tName + ": Dynamic Size to Hit Ratio")
    plt.xlabel('Virtual Time')
    plt.ylabel('Ratio (%)')
    plt.legend()
    """
    Set y axis begin at 0
    """
    plt.ylim(bottom=0)
    try:
        os.mkdir("graph/ratio/" + tName)
    except:
        pass
    plt.savefig("graph/ratio/" + tName + "/" + tName + "_" + size)
    plt.close()


def get_result(path):
    print(path + "..")
    hit_ratio = []
    dynamic_ratio = []
    virtual_time = []
    with open(path, 'r') as f:
        for flt in f.readlines():
            flt = flt.strip().split(',')
            if not flt:
                continue
            virtual_time.append(flt[0])
            hit_ratio.append(flt[1].strip('.'))
            dynamic_ratio.append(flt[2].strip('.'))

        return virtual_time, hit_ratio, dynamic_ratio


if __name__ == "__main__":
    if (len(sys.argv) != 2):
        raise ("Argument Error")
    tName = sys.argv[1]

    with codecs.open("cache_size/" + tName, "r", "UTF8") as inputFile:
        inputFile = inputFile.readlines()
    for line in inputFile:
        line = line.strip()
        line = line.strip("\n")
        virtual_time, hit_ratio, dynamic_ratio = \
            get_result("result_set/" + tName + "/ml_lirs_v8_" + line + "_" + tName + "_ratios")
        plot(virtual_time, hit_ratio, dynamic_ratio, tName, line)