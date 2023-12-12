import collections
import itertools

from scipy.spatial import distance

from wcs_helper_functions import *
from scipy import stats,spatial
import math
from numpy import mean
import pandas
from random import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# read in data
munsellInfo = readChipData('./WCS_data_core/chip.txt')
cielabCoord = readClabData('./WCS_data_core/cnum-vhcm-lab-new.txt')
namingData = open('./WCS_data_core/term.txt','r')
fociData = readFociData('./WCS_data_core/foci-exp.txt')
speakerInfo = readSpeakerData('./WCS_data_core/spkr-lsas.txt')

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


def generate_sim_dict():
    """
    The purpose of this is to pre-generate the similarity values between all chips
    Index at [gloss_chip][speaker_chip] to retrieve the similarity value
    :param namingDict:
    :return: A 2d 330x330 np array containing the similarity score of each index to each index in the space
    """
    similarity_matrix = np.zeros((331, 331))

    for i in range(1, 331):
        for j in range(1, 331):
            sim_result = similarity(i, j)
            similarity_matrix[i][j] = sim_result
            # print(similarity_matrix[i][j])

    return similarity_matrix


def similarity(item1, item2):
    """

    :param item1: Munsell index to compare to
    :param item2: Query Munsell index
    :return: The similarity between the two objects

    """
    # each item is a chip index which has a corresponding Munsell and CIELAB chip
    cielab_1 = np.array(cielabCoord[int(item1)])
    cielab_2 = np.array(cielabCoord[int(item2)])
    c_1 = cielab_1.astype(float)
    c_2 = cielab_2.astype(float)

    dist = spatial.distance.euclidean(c_1, c_2)
    a = -(dist**2)
    return math.exp(a)


def generate_similarity_rankings(namingDict, similarityMatrix):
    """ Using the exemplar model of categorization,
    for each colour gloss of a speaker's language,
    return the similarity ranking of each of the speaker's
    chip indexes to the colour gloss.

    Return a nested dictionary where each speaker corresponds to a dictionary
    where the keys is a colour gloss
    and the value is a list of tuples.
    The tuples contain the chip index and their
    corresponding similarity ranking to that gloss term.
    speaker_sim_ratings = {lang1: {speaker1: {gloss1: [], gloss2: []}}}

    """

    speaker_sim_ratings = {}

    for language in namingDict:
        speaker_sim_ratings[language] = {}

        for speaker in namingDict[language]:
            speaker_sim_ratings[language][speaker] = {}

            for speakers_gloss in namingDict[language][speaker]:  # for each gloss in that speaker's language:
                speaker_sim_ratings[language][speaker][speakers_gloss] = []

                # iterating through all the speaker's chips:
                for gloss_term in namingDict[language][speaker]:
                    for speaker_chip in namingDict[language][speaker][gloss_term]: # get the sum of similarities of this chip to each chip in the glossed term
                        similarity_sum = 0
                        for gloss_chip in namingDict[language][speaker][speakers_gloss]:
                            similarity_sum += similarityMatrix[int(gloss_chip)][int(speaker_chip)]

                        speaker_sim_ratings[language][speaker][speakers_gloss].append((similarity_sum, speaker_chip))

    return speaker_sim_ratings


def generate_rankings(speaker_sim_ratings):
    """
    Given the similarity rankings of each speaker's chip indexes,
    return an ordered ranking of similarities.
    :param speaker_sim_ratings: a dictionary of the format
    {lang1: {speaker1: {gloss1: [(similarity_rating, chip_index),...], gloss2: [(sim_rating, chip_index)]...}}}
    :return: {lang1:{speaker1: {gloss1: [(rating, similarity_rating, chip_index),...], gloss2: [(rating, sim_rating, chip_index)]}}}
    """
    # sort similarity ratings in descending order
    for lang in speaker_sim_ratings:
        for speaker in speaker_sim_ratings[lang]:
            for gloss_term in speaker_sim_ratings[lang][speaker]:
                speaker_sim_ratings[lang][speaker][gloss_term] = sorted(speaker_sim_ratings[lang][speaker][gloss_term], key=lambda x: x[0], reverse=True)

    # generate a corresponding ranking to all the similarities
    sim_to_ratings = {}
    for each_language in speaker_sim_ratings:
        sim_to_ratings[each_language] = {}

        for s in speaker_sim_ratings[each_language]:
            sim_to_ratings[each_language][s] = {}

            for g in speaker_sim_ratings[each_language][s]:
                sim_to_ratings[each_language][s][g] = []

                for index in range(1, len(speaker_sim_ratings[each_language][s][g]) + 1):
                    sim_to_ratings[each_language][s][g].append((index, speaker_sim_ratings[each_language][s][g][index - 1]))

    return sim_to_ratings


def foci_predictions(r):
    """Given a dictionary of ranking data, return a list of all predicted foci.
    Predicted foci calculated by for each language, speaker, gloss category
    get the number of emperical foci, n, given, and choose the top n for that
    languages' speaker's gloss categories' ranked chips as the predicted foci.
    """
    fociPred = []  # list of chips (letter, number) that are predicted foci
    for lang in fociData.keys():  # iterate through languages in empirical foci
        langt = r[str(lang)]  # iterate through languages in ranking data
        for speak in fociData[lang].keys(): # iterate through speakers empirical
            if str(speak) in langt:
                speakt = langt[str(speak)]  # iterate through speakers ranking
                for gloss in fociData[lang][speak].keys():  # iterate through gloss empirical
                    if gloss in speakt:
                        glosst = speakt[gloss]  # iterate through gloss rankings
                        n = len(fociData[lang][speak][gloss])  # count number of foci
                        topN = glosst[:n]  # get top n rankings
                        for chip in topN:
                            fociPred.append(chip)  # append to list
                    else:
                        continue
            else:
                continue
    return fociPred


# get data to plot (add counts)
def add_counts(fp):
    """Given a list of predicted foci, return a list of tuples of all chips
    and the number of times they appear in the predicted foci
    """
    foci_preds = []
    plotData = []  # list to save plot data
    uniqueChips = []  # temp list for all chips
    cnum, cname = munsellInfo
    for i in range(1, 331):  # iterate through all chips
        uniqueChips.append(cname[i])  # put (letter, number) of all chips into list
    for chip in fp:
        foci_preds.append(cname[int(chip[1][1])])


    for chip in uniqueChips:  # iterate through all chips
        count = foci_preds.count(chip)  # get number of appearances in predicted foci
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
    similarity_dict = generate_sim_dict()
    similarities = generate_similarity_rankings(data, similarity_dict)
    ranking_dict = generate_rankings(similarities)  # get ranking dictionary
    predicted_foci = foci_predictions(ranking_dict)  # get list of foci preds
    data_all = add_counts(predicted_foci)

    Y = ["B", "C", "D", "E", "F", "G", "H", "I"]

    X = []
    for i in range(1, 41):
        X.append(i)

    Z = get_contour_z_data(data_all)
    plt.contour(X, Y, Z, [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    plt.show()

    # return get_contour_data(data_all)  # return three arrays

if __name__ == '__main__':

    plot_data(namingDict)


