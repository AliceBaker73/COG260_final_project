import pandas as pd

from wcs_helper_functions import *

import numpy as np
from scipy import stats,spatial
from random import random
import matplotlib
import math
from numpy import mean
from collections import defaultdict

unsellInfo = readChipData('./WCS_data_core/chip.txt')
cielabCoord = readClabData('./WCS_data_core/cnum-vhcm-lab-new.txt')
namingData = open('./WCS_data_core/term.txt','r')
fociData = readFociData('./WCS_data_core/foci-exp.txt')
speakerInfo = readSpeakerData('./WCS_data_core/spkr-lsas.txt')

fociList = []

for lang in fociData.keys():
    for speak in fociData[lang].keys():
        for gloss in fociData[lang][speak].keys():
            fociList.extend(fociData[lang][speak][gloss])


for i in range(len(fociList)):
    fociList[i] = tuple(fociList[i].split(":"))


counts = []  # list to save plot data
uniqueChips = []  # temp list for all chips
cnum, cname = unsellInfo
for i in range(1, 331):  # iterate through all chips
    uniqueChips.append(cname[i])  # put (letter, number) of all chips into list

for chip in uniqueChips:  # iterate through all chips
    count = fociList.count(chip)  # get number of appearances in predicted foci
    counts.append(count)


countData = [counts[i:i+40] for i in range(0, 320, 40)]


def plot_data(data):

    Y = ["B", "C", "D", "E", "F", "G", "H", "I"]

    X = []
    for i in range(1, 41):
        X.append(i)

    Z = data
    plt.contour(X, Y, Z, [100, 200, 300, 400, 500, 600])
    plt.show()


if __name__ == '__main__':
    plot_data(countData)
