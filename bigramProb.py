#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:10:45 2020

"""

import csv
import json
import pickle
import sys
import pdb
import time

def readData(fileName):
	data = []
	file = open(fileName, "r")

	for word in file.read().split():
		data.append(word)

	file.close()
	return data

def createBigramModel(data, START_STRING, file_name='data/bigram.txt'):
    st = time.time()
    listOfBigrams, unigramCounts, bigramCounts = createBigram(data,START_STRING)
    print('create bigram ' + str(time.time()-st))
    st = time.time()  
    listOfProb = calcBigramProb(listOfBigrams, unigramCounts, bigramCounts)
    print('calcBigramProb ' + str(time.time()-st))

    file = open(file_name, 'w')
    for bigrams in listOfBigrams:
        file.write(bigrams[0]+ ' ' + bigrams[1] + ' ' + str(listOfProb[bigrams]) + '\n')
    file.close()
    return listOfProb

def createBigramModel2(data, pos_index, START_STRING, file_name='data/bigram.txt'):
    st = time.time()
    listOfBigrams, unigramCounts, bigramCounts = createBigram(data,pos_index,START_STRING)
    print('create bigram' + str(time.time()-st))
    st = time.time()
    listOfProb = calcBigramProb(listOfBigrams, unigramCounts, bigramCounts)
    print('calcBigramProb' + str(time.time()-st))

    file = open(file_name, 'w')
    for bigrams in listOfBigrams:
        file.write(bigrams[0]+ ' ' + bigrams[1] + ' ' + str(listOfProb[bigrams]) + '\n')
    file.close()
    return listOfProb

def createBigram(data, START_STRING):
    listOfBigrams = set([])
    bigramCounts = {}
    unigramCounts = {}
    for entity in  data:
        for i in range(len(entity)):
            if i > 0:
                previous_word = entity[i - 1]
            else:
                previous_word = START_STRING
                
            listOfBigrams.add((entity[i], previous_word))
            if (entity[i], previous_word) in bigramCounts.keys():
                bigramCounts[(entity[i], previous_word)] += 1
            else:
                bigramCounts[(entity[i], previous_word)] = 1
            if entity[i] in unigramCounts.keys():
                unigramCounts[entity[i]] += 1
            else:
                unigramCounts[entity[i]] = 1

    return listOfBigrams, unigramCounts, bigramCounts

def createBigram2(data, pos_index, START_STRING):
    listOfBigrams = []
    bigramCounts = {}
    unigramCounts = {}
    for pos  in  pos_index:
        entity = data[pos]
        for i in range(len(entity)):
            if i > 0:
                previous_word = entity[i - 1]
            else:
                previous_word = START_STRING
                
            listOfBigrams.append((entity[i], previous_word))
            if (entity[i], previous_word) in bigramCounts:
                bigramCounts[(entity[i], previous_word)] += 1
            else:
                bigramCounts[(entity[i], previous_word)] = 1
            if entity[i] in unigramCounts:
                unigramCounts[entity[i]] += 1
            else:
                unigramCounts[entity[i]] = 1
    return listOfBigrams, unigramCounts, bigramCounts


# ------------------------------ Simple Bigram Model --------------------------------


def calcBigramProb(listOfBigrams, unigramCounts, bigramCounts):

	listOfProb = {}
	for bigram in listOfBigrams:
		word1 = bigram[0]		
		listOfProb[bigram] = (bigramCounts.get(bigram))/(unigramCounts.get(word1))
	return listOfProb


