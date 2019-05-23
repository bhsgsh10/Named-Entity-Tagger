# Named-Entity-Tagger
Tagging words to parts of speech

* Files provided: count_freqs.py, eval_ne_tagger.py, ner_train.dat, ner_dev.dat, ner_dev.key, trigrams.txt
* Python files created: rare_words_replacer.py, base_hmm_tagger.py , q_parameters_calculator.py, viterbi.py, enhanced_viterbi.py
* Text and data files that would be created on running the above files: ner.counts, ner_train_rare, basic_tagged_wordset.txt, viterbi_tagged_wordset.txt, ner_train_rare6, en_viterbi_tagged_wordset.txt, ner_rare6.counts, ner_rare.counts 
## Instructions
There are four types of named entities: person names (PER), organizations (ORG), locations (LOC) and miscellaneous names (MISC). Named entity tags have the format I-TYPE. Only if two phrases of the same type immediately follow each other, the first word of the second phrase will have tag B-TYPE to show that it starts a new entity. Words marked with O are not part of a named entity.
The script count freqs.py reads in a training file and produces trigram, bigram and emission counts. Run the script on the training data and pipe the output into some file: 

	python count freqs.py ner train.dat > ner.counts

There are 4 types of counts in the counts file:
- Count(y~>x): 13 WORDTAG I-ORG University says that the word ‘University’ was tagged 13 times as ‘I-ORG’
- Bigram count: 3792 2-GRAM O I-ORG indicates that there were 3792 instances of an I-ORG following an O tag.
- Trigram count: 1586 3-GRAM O I-ORG I-ORG indicates that in 1586 cases the bigram O I-ORG was followed by another I-ORG tag.
- Unigram count: 11 1-GRAM B-LOC indicates that B-LOC was observed 11 times in the corpus.

Tasks:
1. Replace rare words with _RARE_: There are a number of words in the training dataset which occur less than 5 times. Using rare_words_replacer.py, we replace all those words with the string ‘_RARE_’. Run this file using the following command:   

    `python3 rare_words_replacer.py`

2. Writing a basic HMM tagger: The base_hmm_tagger.py calculates emission probabilities based on the following formula: e(x|y) = Count(y~>x) / Count(y) and uses the probabilities to predict tags for words. Run this file using the following command:   
  
    `python3 base_hmm_tagger.py`

  Run the evaluation of your tagger using the following command:      
	```
    	python eval_ne_tagger.py ner_dev.key basic_tagged_wordset.txt
    	```   

Total F1-score for this tagger should be more than 0.3.

3. Compute maximum likelihood parameters for trigrams in the training data: This operation is performed by q_parameters_calculator.py. Run this file using the following command:   
  
    `python3 q_parameters_calculator.py`
  
4. Implementation of Viterbi algorithm for part of speech tagging: This task is available in viterbi.py. Run this file using the following command:   

    `python3 viterbi.py`
  
  Run the evaluation of your tagger using the following command: 	   
    	`python eval_ne_tagger.py ner_dev.key viterbi_tagged_wordset.txt`
    
 Total F1-score for this tagger should be more than 0.67. 
  
5. Enhanced Viterbi: Instead of replacing all infrequent words with _RARE_, group them into different classes such as infrequent words starting with capital letters, or words containing numbers. Choose classes of your own liking and see the changes in accuracy of tagging. Re-run Viterbi algorithm with the new classes of rare words. This task is available on enhanced_viterbi.py. Following classification has been used for rare words:   
  * Numeric: when word contains only numbers
  * Last caps: when last letter of the word is in caps
  * All caps: when the entire word is in caps
  * All small: when the entire word is in small letters
  * Rare: if the infrequent word does not fall into any of the above categories, then it is classified 	as 'RARE'   
 
 Run this file using the following command:   
  
      `python3 enhanced_viterbi.py`

 Run the evaluation of your tagger using the following command:    
  	`python eval_ne_tagger.py ner_dev.key en_viterbi_tagged_wordset.txt`

Total F1-score for this tagger should be more than 0.72.
