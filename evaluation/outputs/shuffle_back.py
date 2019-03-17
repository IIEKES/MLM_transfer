import math

from nltk import word_tokenize
from collections import Counter
from nltk.util import ngrams
import sys
import string

def shffule_back(generate_file, orgin_file):
    fro = open(orgin_file, "r")
    frg = open(generate_file, "r")
    fw = open("{}.gen".format(generate_file), 'w')
    index = []
    for line in fro:
        splits = line.lower().strip().split('\t')
        index.append(splits[0])
    fro.close()
    cand = []
    for line in frg:
        cand.append(line)
    frg.close()
    for id in index:
        for cd in cand:
            if id in cd:
                fw.write(cd)
                break
    fw.close()

if __name__ == "__main__":

    generate_file = "yelp/sentiment.test.0.cbert"
    orgin_file = "yelp/sentiment.test.0.human"
    shffule_back(generate_file, orgin_file)

    generate_file = "yelp/sentiment.test.1.cbert"
    orgin_file = "yelp/sentiment.test.1.human"
    shffule_back(generate_file, orgin_file)
