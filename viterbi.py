#!/usr/bin/python3


'''
Found 4702 NEs. Expected 5931 NEs; Correct: 3646.

	     precision 	 recall 	F1-Score
Total:	 0.775415	0.614736	0.685790
PER:	 0.763231	0.596300	0.669517
ORG:	 0.611855	0.478326	0.536913
LOC:	 0.876289	0.695202	0.775312
MISC:	 0.830065	0.689468	0.753262
'''

import sys
import math
from collections import defaultdict
import base_hmm_tagger
import q_parameters_calculator

large_negative_value = -9999999999999999


def get_possible_tags():

    '''
    :return: all possible tags for positions 1 to n
    '''
    all_possible_tags = []
    # read the unigrams from the counts file
    fp = open('ner_rare.counts')
    file_contents = fp.read()
    all_lines = file_contents.split('\n')
    for line in all_lines:
        if "1-GRAM" in line:
            words = line.split(' ')
            tag = words[2]
            all_possible_tags.append(tag)

    fp.close()

    return all_possible_tags



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

def get_emission_probability(word, tag, emissions, possible_tags):
    '''
    Returns emission probability for the given word tag pair
    :param word: word of the sentence
    :param tag: the tag for which we need to check emission probability
    :param emissions: all emission probabilities calculated using the counts file
    :return: emission probability for the given word, tag pair
    '''
    e_prob = large_negative_value
    if tag in emissions[word]:
        e_prob = math.log2(emissions[word][tag])
    else:
        # loop over all possible tags for this word in emissions
        # if word does not have any tags, then get emission for _RARE_
        # because in that case, the word is infrequent
        set_rare = True
        for possible_tag in possible_tags:
            if possible_tag in emissions[word]:
                set_rare = False
                break
        if set_rare == True:
            if tag in emissions['_RARE_']:
                e_prob = math.log2(emissions['_RARE_'][tag])

    return e_prob

def get_q_param(trigram):
    '''
    Returns the value of q_parameter for the given trigram
    :param trigram: the trigram for which we need the q_param value
    :return: value of q_parameter for the given trigram
    '''
    q_param = large_negative_value
    if trigram in q_params:
        q_param = math.log2(q_params[trigram])

    return q_param


def viterbi(sentence, q_params, emission, tags_file):
    '''
    The implemetation of Viterbi algorithm
    :param sentence: a list of words in asingle sentence
    :param q_params: the dictionary of all q parameters
    :param emission: the dictionary of all emission parameters
    :param tags_file: the list of possible tags identified in the counts file
    :return: the tag sequence for the input sentence
    '''
    pi = {}
    back_pointers = {}
    pi[(0, '*', '*')] = 0

    n = len(sentence)

    tag_dict = {}
    tag_dict[-1] = ['*']
    tag_dict[0] = ['*']
    for i in range(n):
        tag_dict[i + 1] = tags_file

    for k in range(0, n):
        for u in tag_dict[k]:
            for v in tag_dict[k+1]:
                # print((k, u, v, sentence[k]))
                # declare a list that will hold all the sums of probability + q_parameter + emission_probability
                pi_sum = []
                for w in tag_dict[k-1]:
                    # declare a variable to hold probability value
                    p = pi[(k, w, u)]

                    # set q_parameter using q_params
                    trigram = w + ' ' + u + ' ' + v
                    q_param = get_q_param(trigram)

                    # set emission probability using emission probabilities
                    word = sentence[k]
                    e_prob = get_emission_probability(word, v, emission, tags_file)

                    pi_sum.append(p + q_param + e_prob)

                pi[(k+1, u, v)] = max(pi_sum)
                # the backpointer point to that tag for which we got the max value of pi
                back_pointers[(k+1, u, v)] = tag_dict[k-1][pi_sum.index(pi[(k+1, u, v)])]

    '''
    compute the final two tags for the sequence.
    '''
    y = calculate_last_two_tags(tag_dict, pi, q_params, n)

    '''
    Now that we have the final 2 tags, we use the backpointers to generate rest of the sequence
    '''

    for i in range(n-2, 0, -1):
        y_list = y[:]
        key = (i+2, y_list[0], y_list[1])
        if key in back_pointers:
            y = [back_pointers[key]] + y_list

    return y

def calculate_last_two_tags(tag_dict, pi, q_params, n):
    '''
    Implementation to return last two tags for the tag sequence
    :param tag_dict: dictionary of possible tags
    :param pi: all pi values pi[(k, u, v)]
    :param q_params: all q_parameters
    :param n: length of the sentence
    :return: last two tags of the tag sequence
    '''
    max_pi, max_u, max_v = -math.inf, None, None

    for u in tag_dict[n - 1]:
        for v in tag_dict[n]:
            trigram = u + ' ' + v + ' ' + 'STOP'
            if trigram in q_params:
                pi_sum = pi[(n, u, v)] + math.log2(q_params[trigram])
                if pi_sum > max_pi:
                    max_pi = pi_sum
                    max_u = u
                    max_v = v

    return [max_u, max_v]


def tagger(input_file, q_params, emissions, output_file, tags_file):
    '''
    This function parses the input file line by line and collects words, puts them in a list, and calls the Viterbi
    algorithm when it encounters a blank line (end of a sentence). It repeats this process for each sentence.
    It receives a tag sequence each time it calls Viterbi. It then iterates over the words of each sentence
    and pairs it up with the corresponding tag from the tag sequence. Then it computes the sum of log2
    probability for the word tag pairs upto that point. Then it writes word, tag, probability tuple to the output file
    :param input_file: the file used to read the data from. This is the dev file
    :param q_params: the dictionary of all q parameters estimated from the counts file
    :param emissions: the dictionary of emission probabilities estimated from the counts file
    :param output_file: the file we write to
    :param tags_file: the list of possible tags as observed in the counts file
    '''
    fp = open(input_file)
    file_contents = fp.read()
    fp.close()

    all_lines = file_contents.split('\n')
    sentence_words = []
    new_lines = []
    sentences = []
    for line in all_lines:
        # collect all words till we reach an empty line. At that point we call Viterbi
        if line == '':
            new_lines.append(line)
            tag_sequence = viterbi(sentence_words, q_params, emissions, tags_file)
            sum_probability = 0
            #print(tag_sequence)
            for item in range(len(sentence_words)):
                probability = get_emission_probability(sentence_words[item], tag_sequence[item], emissions, tags_file)
                sum_probability += probability
                new_line = ' '.join(map(str, (sentence_words[item], tag_sequence[item], float(sum_probability)))) + '\n'
                new_lines.append(new_line)

            sentence_words = []
        else:
            sentence_words.extend(line.split(' '))

    index = 0
    fo = open(output_file, 'w')
    for new_line in new_lines:
        if new_line == '':
            # the following check prevents a new line character to be appended at the beginning of the file
            if index > 0:
                fo.write('\n')
        else:
            fo.write(new_line)
        index += 1

    fo.close()

if __name__ == "__main__":
    tags_in_file = get_possible_tags()
    # get the q_params from the counts file
    q_params = q_parameters_calculator.get_trigram_probabilities('ner_rare.counts')
    emission_probabilities = calculate_emission_probabilities('ner_rare.counts')
    tagger('ner_dev.dat', q_params, emission_probabilities, 'viterbi_tagged_wordset.txt', tags_in_file)
