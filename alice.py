import collections
import copy
import numpy as np
from operator import itemgetter

class Alice():
    def __init__(self):
        self.start_symbol = "<SOS>"
        self.end_symbol = "<EOS>"
        self.start_word_symbol = "<SOW>"
        self.end_word_symbol = "<EOW>"

        self.alice_data = []
        self.alice_dence = []
        self.alice_test = []
        self.alice_train = []
        self.keylist = []
        self.charlist = []

        load_data = open("alice_formatted.txt", "r")
        for line in load_data:
            self.alice_data.append(line.split())
            for elem in line.split():
                self.alice_dence.append(elem)

        load_data.close()

        load_data = open("alice_test.txt", "r")
        for line in load_data:
            self.alice_test.append(line.split())

        load_data.close()

        load_data = open("alice_train.txt", "r")
        for line in load_data:
            self.alice_train.append(line.split())

        load_data.close()

        self.collection = collections.Counter(self.alice_dence)
        self.collection[self.start_symbol] = 1
        self.collection[self.end_symbol] = 1
        self.collection[self.start_word_symbol] = 1 # 追加
        #self.collection[self.end_word_symbol] = 1 # 追加
        self.keylist = list(self.collection.keys())

        self.charlist = self.calc_num_char(self.alice_dence)

    def get_origin_data(self):
        return self.alice_data

    def get_char_ngram(self, word, n):
        char_list = []
        char_list = copy.copy(word) # ここが値参照でバグっていた

        #print("before_seq: ", char_list)

        for i in range(n - 1):
            char_list.insert(0, self.start_word_symbol)
        #char_list.append(self.end_word_symbol)

        #print("after_seq: ", list(zip(*(char_list[i:] for i in range(n)))))

        return list(zip(*(char_list[i:] for i in range(n))))

    def get_ngram(self, sentence, n):
        seq = []
        seq = copy.copy(sentence) # ここが値参照でバグっていた

        for i in range(n - 1):
            seq.insert(0, self.start_symbol)
        seq.append(self.end_symbol)

        return list(zip(*(seq[i:] for i in range(n))))

    # [sentence, sentence, sentence]
    def get_ngrams(self, sentences, n):
        ngrams = []
        for sentence in sentences:
            ngrams.append(self.get_ngram(sentence, n))
        return ngrams

    def get_validation_data(self):
        return self.alice_test, self.alice_train

    def get_vocabulary(self):
        #print("self.keylist: ", self.keylist)
        return self.keylist

    def calc_num_char(self, word_list):
        char_list = []
        for word in word_list:
            for char in word:
                char_list.append(char)

        char_collection = collections.Counter(char_list)
        char_collection[self.start_word_symbol] = 1
        #char_collection[self.end_word_symbol] = 1 # 追加
        char_collection[self.start_symbol] = 1 # 追加
        char_collection[self.end_symbol] = 1 # 追加
        return list(char_collection.keys())

    def get_chars(self):
        return self.charlist

    def get_num_char(self):
        return len(self.charlist)

    def get_num_vocabulary(self):
        return len(self.collection)
