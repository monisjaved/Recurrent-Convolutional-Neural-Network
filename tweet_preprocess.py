# -*- coding: utf-8 -*-
"""
Pre-processing Tool for Tweets
v0.1
Author: Monis Javed
Creation Date: 12th December 2015
Last Updated on: 24th December 2015
------------------------------------------------------------------------------------------------------

PREREQUISITE LIBRARIES:
'twokenize'         (for tokenizing tweets )
'csv'               (for loading and saving data)
'word2vec'          (for generating initial vector for training)
'numpy'             (for processing arrays)
'random'            (for shuffling)

-----------
DESCRIPTION
-----------
The class TweetPreProcess performs all the per-processing of the
data. This class contains functions to generate the initial word
embeddings from the text corpus using word2vec.

----------------
EXAMPLE USE-CASES
-----------------

t = TweetPreProcess()
x, y = t.create_set(filename="data/1700.csv", random=True)

vector_dict = theano.shared(np.asarray(t.word_vec).astype("float32"), name="vector_dict")

-----------
DATA USED
-----------
For generating vectors, the 10K database has been used resulting in around 2500
vectors as output. All other words will be marked as UNKNOWN.
"""
import csv
import twokenize
import word2vec
import numpy as np
from random import shuffle

filename = "data/twitter-tagged-full.csv"


class TweetPreProcess:
    # model = None
    @staticmethod
    def tokenize(sentence):
        """
        Uses twokenize to tokenize tweets.

        :param sentence: String of the tweet
        :return: list containing the tokens
        """
        toks = twokenize.tokenizeRawTweetText(sentence.lower())
        return toks

    def __init__(self, file_name="data/training_vec.bin", vector_size=100):

        """
        Initializes the word2vec object and create dictionaries for any type of conversions
        :param file_name: string of the filename to load binary file of vectors created by word2vec
        :param vector_size: the size of vector of the word embedding
        """
        self.model = word2vec.load(file_name)
        self.word_list = self.model.vocab.tolist()
        self.word_list.append("UNK")
        self.word_index = {}

        for i, word in enumerate(self.word_list):
            self.word_index[word] = i

        self.word_vec = []
        for x in self.word_list[:-1]:
            self.word_vec.append(self.model[x])

        self.word_vec.append(np.random.randn(vector_size))  # Vector for random

    def sentence2index(self, sentence, word_index=None):
        """
        Converts sentence to list of indexes representing the words and their corresponding vectors
        :param sentence: Sentence to convert
        :param word_index: Define the dictionary from word to their indexes
            if not specified automatically takes one from the object
        :return: list of indexes of word
        """
        if word_index is None:
            word_index = self.word_index
        toks = self.tokenize(sentence)
        index = []
        for token in toks:
            try:
                i = word_index[token]
            except KeyError:
                i = len(word_index) - 1  # Last element is unknown

            index.append(i)
        index.append(word_index["</s>"])
        return index

    def index2sentence(self, arr, word_list=None):
        """
        Given indexes outputs the space separated list of tokens
        :param arr: Array of indexes
        :param word_list: List of the words at the same index as in the word_index
        :return: string representing the tweet
        """
        if word_list is None: word_list = self.word_list
        toks = []
        for x in arr:
            toks.append(word_list[x])
        if toks[-1] == "</s>": toks.pop()
        return " ".join(toks)

    def create_set(self, filename, random=False):
        # Each row format as [Text, int, int, int, int]
        """
        Converts csv data set to variables to be used by RCNN.

        The csv data format should be
        <Tweet>, class1,class2,class3...

        Converts tweet to its corresponding indexes list

        :param filename: string of file path
        :param random: boolean, if true shuffle data and then load
        :returns: (x, y)
            :return x: list of list representing indexes of
            :return y: 2D list of output values
        """
        sentences = []
        output = []
        with open(filename, "rb") as fin:
            csv_read = csv.reader(fin)
            for row in csv_read:
                sentences.append(row[0])
                o = [float(x) for x in row[1:]]
                output.append(o)

        t_set = []
        for s in sentences:
            t_set.append(self.sentence2index(s))

        if random:
            order = range(len(t_set))
            shuffle(order)
            t_set = [t_set[x] for x in order]
            output = [output[x] for x in order]
        return t_set, output


# Use this code to convert corpse to vectors
# word2vec.word2vec('data/tweet-corpse.txt', 'data/training_vec.bin', size=100, verbose=True)


