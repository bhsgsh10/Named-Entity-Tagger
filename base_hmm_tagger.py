#!/usr/bin/python3
import sys
import math
from collections import defaultdict


'''
Found 14043 NEs. Expected 5931 NEs; Correct: 3117.

	     precision 	 recall     F1-Score
Total:	 0.221961	0.525544	0.312106
PER:	 0.435451	0.231230	0.302061
ORG:	 0.475936	0.399103	0.434146
LOC:	 0.147750	0.870229	0.252612
MISC:	 0.491689	0.610206	0.544574
'''

def calculate_emission_probabilities(filename):
    '''
    Calculates emission probailities for each word given the tags it appears with in
    the document.
    :param filename:  'ner_rare.counts'
    :return: dictionary of emission probabilities
    '''

    emission_probabilities = defaultdict(dict)
    fp = open(filename)

    file_contents = fp.read()
    # segregate all those lines which have WORDTAG from the ones that don't
    wordtag_sentences = []
    # store all 1-gram sentences separately
    one_gram_sentences = []
    all_lines = file_contents.split('\n')
    for line in all_lines:
        # if WORDTAG is found in a sentence, put it in wordtag_sentences,
        # else if the line contains "1-GRAM" put it into one_gram_sentences
        if "WORDTAG" in line:
            wordtag_sentences.append(line)
        elif "1-GRAM" in line:
            one_gram_sentences.append(line)

    # create a dictionary that would store all the unique tags and their counts
    # do this by looping over the one_gram_sentences
    one_gram_map = {}
    for one_gram_sentence in one_gram_sentences:
        # get the first and the last words from each sentence
        # the first word gives the count and the last word gives the tag
        words_onegram = one_gram_sentence.split(' ')
        count = int(words_onegram[0])
        tag = words_onegram[-1]
        one_gram_map[tag] = count

    # iterate over all wordtag_sentences and calculate emission probabilities
    for wordtag_sentence in wordtag_sentences:
        # split the sentence into words
        words = wordtag_sentence.split(' ')
        # get the count
        count_xy = int(words[0])
        # get the word
        x = words[-1]
        # get the tag
        y = words[-2]

        # now find the count for y in one_gram_map.
        #  if the quantity is not found or if it 0 by any chance, then the resulting
        # emission probability should be set to 0
        emission_prob = 0
        count_y = 1
        if y in one_gram_map:
            count_y = one_gram_map[y]
        if count_y != 0:
            emission_prob = float(count_xy)/float(count_y)

        emission_probabilities[x][y] = emission_prob

    fp.close()

    return emission_probabilities

def tagger(counts_file, dev_file, output_file):
    '''
    Implementation of the basic tagger that reads in the counts file and the dev file and tags words in the dev
    file. This tagger does not take into account context, ie. what occurred in the sentence prior to the current
    word.
    :param counts_file: ner_rare.counts
    :param dev_file: ner_dev.dat
    :param output_file: the file we write to, '4_2.txt'
    '''

    emission_probabilities = calculate_emission_probabilities(counts_file)
    fd = open(dev_file, 'r')
    fo = open(output_file, 'w')
    for line in fd:
        word = line.strip()
        if len(word) < 1:
            fo.write(line)
            continue

        word_tag_dict = {}
        # if word exists in emission_probabilities, then we pick the tag, probability pairs from
        # emission_probabilities[word] else from emission_probabilities['_RARE_']
        if word in emission_probabilities:
            word_tag_dict = emission_probabilities[word]

        else:
            word_tag_dict = emission_probabilities['_RARE_']


        max_probability = max(word_tag_dict.values())
        max_tag = [tag for tag, pr in word_tag_dict.items() if pr == max_probability][0]
        log_probability = math.log2(max_probability)
        new_tuple = (word, max_tag, log_probability)
        # form the line to be written to the output file
        new_line = ' '.join(map(str, new_tuple)) + '\n'

        fo.write(new_line)


    fd.close()
    fo.close()


if __name__ == "__main__":
    tagger('ner_rare.counts', 'ner_dev.dat', 'basic_tagged_wordset.txt')
