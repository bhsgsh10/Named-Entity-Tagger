#! /usr/bin/python3

import sys


def replace_with_rare(filename):
    '''
    This function takes in a file and replaces all those words with a frquency of less than 5,
    with a single word: _RARE_
    :param filename: 'ner_train.dat'
    :return: modified contents of the file. Infrequent words are replaced with _RARE_
    '''
    fp = open(filename)
    file_contents = fp.read()
    fp.close()
    # read line by line and collect all the words.
    # We can do this in a single loop. As we iterate, we keep adding words into a dictionary
    # with count initialized to 1. We increment the count if we encounter the same word again.
    # From that dictionary, we can find all those words whose frequency is less than 5.
    # Replace all those words in file_contents with _RARE_.
    word_count_map = {}
    all_lines = file_contents.split('\n')
    for line in all_lines:
        # split line into words
        words = line.split(' ')
        # pick the first word from each line if line is not empty
        if len(words) > 0:
            word = words[0]
            if word in word_count_map.keys():
                # if word already exists in word_count_map then increment its count by 1
                word_count_map[word] += 1
            else:
                # else initialize count to 1
                word_count_map[word] = 1

    new_file_contents = ''

    # iterate over all the lines of the existing file contents
    # replace the words which fall into infrequent_words with _RARE_
    for line in all_lines:
        if len(line) > 0:
            words = line.split(' ')
            if len(words) > 1:
                word = words[0]

                if word_count_map[word] < 5:
                    words[0] = '_RARE_'
                    line = ' '.join(words)

        new_file_contents += line + '\n'

    return new_file_contents


def write_file(input_filename, output_filename):
    '''
    Writes contents received from replace_with_rare() to a new file.
    :param input_filename: 'ner_train.dat'
    :param output_filename: ner_train_rare.dat

    '''
    # get modified content using replace_with_rare()
    new_contents = replace_with_rare(input_filename)
    # create new data file for writing
    fp = open(output_filename, 'w')
    fp.write(new_contents)

if __name__ == "__main__":

    write_file('ner_train.dat', 'ner_train_rare.dat')
