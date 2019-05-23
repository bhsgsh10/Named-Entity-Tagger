#!/usr/bin/python3

import sys
import math

large_negative_value = -9999999999999999

def get_trigram_probabilities(input_file):
    q_params = {}
    fp = open(input_file)
    file_contents = fp.read()
    # create 2 dictionaries, one for trigrams and the other for bigrams
    # <key, value> would be <trigram: count> and <bigram: count> respectively
    trigram_map = {}
    bigram_map = {}
    all_lines = file_contents.split('\n')

    for line in all_lines:
        if "3-GRAM" in line:
            # obtain the trigram and its count from line and put them in trigram_map
            mapping = get_gram_count_mapping(line, " 3-GRAM ")
            trigram_map[mapping[0]] = int(mapping[1])

        elif "2-GRAM" in line:
            # obtain the bigram and its count from line and put them in bigram_map
            mapping = get_gram_count_mapping(line, " 2-GRAM ")
            bigram_map[mapping[0]] = int(mapping[1])

    # to calculate probabilities, we iterate over all keys of trigrams in
    # trigram_map. For each trigram, we search the corresponding bigram in
    # bigram_map. By this point we already know the counts for all the bigrams
    # and trigrams. So we can use those values from the mappings to compute the
    # probabilities

    trigrams = [k for k, v in trigram_map.items()]
    bigrams = [k for k, v in bigram_map.items()]
    for trigram in trigrams:
        bigram = trigram.rsplit(' ', 1)[0]
        count_trigram = trigram_map[trigram]
        count_bigram = bigram_map[bigram]
        probability = float(count_trigram) / float(count_bigram)
        q_params[trigram] = probability

    return q_params



def predict_probabilities(input_file, output_file, counts_file):

    q_params = get_trigram_probabilities(counts_file)

    fp = open(input_file)
    file_contents = fp.read()
    fo = open(output_file, 'w')

    # each line of the input_file contains a trigram. We need to utilize the trigram probabilities
    # calculated in the previous function to predict the probabilities
    all_lines = file_contents.split('\n')
    for line in all_lines:
        if line in q_params:
            log_prob = 1
            if q_params[line] == 0.0:
                log_prob = large_negative_value
            else:
                log_prob = math.log2(q_params[line])
            new_line = line + ' ' + str(log_prob) + '\n'
            fo.write(new_line)

    fo.close()
    fp.close()




def get_gram_count_mapping(line, split_string):
    gram_parts = line.split(split_string)
    count = gram_parts[0]
    gram = gram_parts[1]
    return (gram, count)


if __name__ == "__main__":
    predict_probabilities('trigrams.txt', 'trigrams_q_params.txt', 'ner_rare.counts')