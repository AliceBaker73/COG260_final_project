import pandas as pd

from wcs_helper_functions import *

import numpy as np
from scipy import stats,spatial
from random import random
import matplotlib
import math
from numpy import mean
from collections import defaultdict

# read in data
unsellInfo = readChipData('./WCS_data_core/chip.txt')
cielabCoord = readClabData('./WCS_data_core/cnum-vhcm-lab-new.txt')
namingData = open('./WCS_data_core/term.txt','r')
fociData = readFociData('./WCS_data_core/foci-exp.txt')
speakerInfo = readSpeakerData('./WCS_data_core/spkr-lsas.txt')


# sorting the data for prediction
# loop through rows of term.txt
# # if the language isn't a key, add it, if it is look at the dictionary value
# # if the speaker isn't a key add it, if it is look at the dictionary value
# # if the gloss isn't a key, add it, if if is, add the chip index to a list of
# # chip indicies for that gloss

namingDict = {}
# {language1: {speaker1: {gloss1: [chip_index1, chip_index2...], gloss2: [chip_index9, chip_index6]}, speaker2...}}
for line in namingData:
    line = line.split()
    if line[0] in namingDict:
        if line[1] in namingDict[line[0]]:
            if line[3] in namingDict[line[0]][line[1]]:
                namingDict[line[0]][line[1]][line[3]].append(line[2])
            else:
                if line[3] == "*":
                    pass
                else:
                    namingDict[line[0]][line[1]][line[3]] = [line[2]]
        else:
            namingDict[line[0]][line[1]] = {line[3]: [line[2]]}
    else:
        namingDict[line[0]] = {line[1]: {line[3]: [line[2]]}}


# helper functions to calculate prototype functions
def prototype(l, s, g):
    """Given a language, speaker, and gloss category, return the prototype
    (centeroid) of that category.
    Get the coordinates of every chip in the gloss category and calculate the
    mean of the coordinates, this is the prototype.
    """
    chips = namingDict[l][s][g]  # list of chips
    coords = []
    for c in chips:  # get the coords of all chips and save in list
        coords.append(cielabCoord[int(c)])
    ct = tuple(coords)
    ct = (tuple(tuple(map(float, x)) for x in ct)) # convert strings to floats
    p = list(tuple(map(mean, zip(*ct))))
    return p  # return the mean (prototype)


def distance(c, p):
    """Given chip coordinates and a prototype (coordinates) return the distance
    between the chip and the prototype.
    """

    chip = [float(i) for i in c]  # convert coordinates to floats
    dist = spatial.distance.euclidean(chip, p) # calculate distance
    sim = math.exp(-(dist**2))
    return sim


def it_gloss_ranks(l, s, c, rd):
    for gloss in c:  # iterate through categories/gloss
        ranks_dict = {"chips": [], "ranks": []}
        proto = prototype(l, s, gloss)  # calculate the prototype for this gloss
        # now need to calculate distance from prototype to all chips
        cnum, cname = unsellInfo
        chips = [cname[i] for i in range(1, 331)]
        dists = [distance(list(cielabCoord[i]), proto) for i in range(1, 331)]
        ranks_dict["chips"] = chips
        ranks_dict["ranks"] = dists
        ranks = pd.DataFrame.from_dict(ranks_dict)
        ranks.sort_values('ranks')  # sort values by distance
        if l in rd:  # if language not in dictionary, add it
            if s in rd[l]: # if speaker not in dict, add
                rd[l][s][gloss] = ranks
            else:
                rd[l][s] = {gloss: ranks}
        else:
            rd[l] = {s: {gloss: ranks}}


def it_speak_ranks(l, s, n, rd):
    for speak in s:
        categories = n[l][speak].keys()
        it_gloss_ranks(l, speak, categories, rd)


def it_lang_ranks(l, n, rd):
    for lang in l:
        speakers = n[lang].keys()
        it_speak_ranks(lang, speakers, n, rd)


# calculating rankings - this should get sorted rankings
# {language1 : {speaker1 : {gloss1: [rankings dataframe]}, speaker2 : {gloss1 : [rankings dataframe]} ,...},...}
def get_rankings(n):
    """Given a dictionary of chip naming data, iterate through each language,
    speaker, and gloss category and use prototype to calculate ranking
    (closeness to prototype) of every chip.
    Store the rankings in a dataframe with chip and ranking.
    Chips are stored in munsell format (letter, number)
    """
    # Initialize rankingDict as a nested defaultdict
    rankingDict = defaultdict(lambda: defaultdict(dict))
    languages = n.keys()
    for lang in languages:
        for speak in n[lang].keys():
            for gloss in n[lang][speak].keys():
                proto = prototype(lang, speak, gloss)

                # Extract chips and distances using list comprehensions
                cnum, cname = unsellInfo
                chips = [cname[i] for i in range(1, 331)]
                dists = [distance(list(cielabCoord[i]), proto) for i in range(1, 331)]

                # Sort distances in-place and assign back to ranks
                ranks_dict = {"chips": chips, "ranks": dists}
                ranks = pd.DataFrame(ranks_dict)
                ranks.sort_values('ranks', ascending=False, inplace=True)

                # Update rankingDict using defaultdict
                rankingDict[lang][speak][gloss] = ranks

    # Convert rankingDict to a regular dictionary if needed
    rankingDict = dict(rankingDict)
    return rankingDict


# matching to foci data to get predicted foci
def foci_predictions(r):
    """Given a dictionary of ranking data, return a list of all predicted foci.
    Predicted foci calculated by for each language, speaker, gloss category
    get the number of emperical foci, n, given, and choose the top n for that
    languages' speaker's gloss categories' ranked chips as the predicted foci.
    """
    fociPred = []  # list of chips (letter, number) that are predicted foci
    for lang in fociData.keys():  # iterate through languages in empirical foci
        langt = r[str(lang)]  # iterate through languages in ranking data
        for speak in fociData[lang].keys():  # iterate through speakers empirical
            speakt = langt[str(speak)]  # iterate through speakers ranking
            for gloss in fociData[lang][speak].keys():  # iterate through gloss empirical
                if gloss in speakt:
                    glosst = speakt[gloss]  # iterate through gloss rankings
                    n = len(fociData[lang][speak][gloss])  # count number of foci
                    chips = glosst["chips"].tolist()  # turn column into list
                    topN = chips[:n]  # get top n rankings
                    for chip in topN:
                        fociPred.append(chip)  # append to list
                else:
                    continue
    return fociPred


# get data to plot (add counts)
def add_counts(fp):
    """Given a list of predicted foci, return a list of tuples of all chips
    and the number of times they appear in the predicted foci
    """
    plotData = []  # list to save plot data
    uniqueChips = []  # temp list for all chips
    cnum, cname = unsellInfo
    for i in range(1, 331):  # iterate through all chips
        uniqueChips.append(cname[i])  # put (letter, number) of all chips into list

    for chip in uniqueChips:  # iterate through all chips
        count = fp.count(chip)  # get number of appearances in predicted foci
        data = (chip[0], chip[1], count)  # new tuple (letter, number, count)
        plotData.append(data)
    return plotData


def get_contour_z_data(pd):
    """Given a list of plot data, split tuples into three lists"""
    zProtoData = []
    for data in pd:
        if data[1] == "0" or data[0] == "J" or data[0] == "A":
            continue
        else:
            zProtoData.append(data[2])

    zProtoData2 = [zProtoData[i:i+40] for i in range(0, 320, 40)]
    return zProtoData2


def plot_data(data):
    ranking_dict = get_rankings(data)  # get ranking dictionary
    predicted_foci = foci_predictions(ranking_dict)  # get list of foci preds
    data_all = add_counts(predicted_foci)

    Y = ["B", "C", "D", "E", "F", "G", "H", "I"]

    X = []
    for i in range(1, 41):
        X.append(i)

    Z = get_contour_z_data(data_all)
    plt.contour(X, Y, Z, [100, 200, 300, 400, 500, 600])
    plt.show()

    # return get_contour_data(data_all)  # return three arrays


if __name__ == '__main__':
    plot_data(namingDict)

