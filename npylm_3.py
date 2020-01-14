# -*- coding: utf-8 -*-
"""
2019/1/10 NPYLM (文字n-gramの追加)
"""
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
from pyp import PYP, PYP_prior
from alice import Alice
import pickle
import math
alice = Alice()
char_n_global = 3
#seq = ["A","B","A","B","C","D","A","B","A","B","C","D","C","D","A","B","A","B","A","B","C","D","C","D","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","C","D","C","D","C","D","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","C","D","C","D","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","C","D","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","C","D","C","D","B","A","B","A","B","A","C","D","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B"]

#start_symbol = "<SOS>"
#start_symbol = alice.start_symbol
#end_symbol = "<EOS>"
#end_symbol = alice.end_symbol
#seq = ["A","A","A","B","A","A","A","A","B","A","A","A","A","B","A","A"]

class G0(): #一様分布
    def __init__(self, V):
        self.V = V
        self.num_customers = V
        self.context = ()
        self.n=0
        self.context_restaurant = {}
        self.is_char = True

    def choose_add_table(self, word):
        #print(word)
        pass

    def choose_remove_table(self, word):
        pass

    def word_probability(self, word):
        return 1.0/self.V

    def sample_hyperparameters(self):
        pass
    def check_customer(self):
        pass

    def print_log(self):
        print("is_char: ", self.is_char)
        print("w_hpylm.n: ", self.n)
        print(self)

class w_G0(): # 単語グラムのG0, 文字グラムから単語の生起確率をひく
    def __init__(self, char_n, c_npylm):
        #self.V = V
        #self.num_customers = V
        self.context = ()
        self.context_restaurant = {}
        #self.npylm = HPYLM(n, is_char = True)
        self.n = 0
        self.char_n = char_n
        self.is_char = False
        self.c_npylm = c_npylm

    def choose_add_table(self, word):
        char_list = self.word2charlist(word)
        for char_ngram in alice.get_char_ngram(char_list, self.char_n): # charのn-gram
            char_context = char_ngram[:-1]
            self.c_npylm.choose_add_table(char_context, char_ngram[-1])

    def choose_remove_table(self, word):
        char_list = self.word2charlist(word)
        for char_ngram in alice.get_char_ngram(char_list, self.char_n): # charのn-gram
            char_context = char_ngram[:-1]
            self.c_npylm.choose_remove_table(char_context, char_ngram[-1])

    def word_probability(self, word):
        probability = 1
        #print("word: ", word)
        #print("alice.start_symbol: ", alice.start_symbol)
        char_list = self.word2charlist(word)

        #print("char_list: ", char_list)
        for char_ngram in alice.get_char_ngram(char_list, self.char_n): # charのn-gram
            #print(char_ngram)
            char_context = char_ngram[:-1]
            probability *= self.c_npylm.word_probability(char_context, char_ngram[-1])

        #print("word: ", word)
        #print("len: ", len(char_list))
        #print("poisson: ", self.poisson(lamb=4, k = len(char_list)))
        #print("prob: ", probability * self.poisson(lamb=4, k = len(char_list)))

        #return 1 / alice.get_num_vocabulary()
        return probability * self.poisson(lamb=4, k = len(char_list))
        #self.npylm.word_probability()

    def word2charlist(self, word):
        char_list = []
        if word == alice.start_symbol:
            char_list.append(alice.start_symbol)
        elif word == alice.end_symbol:
            char_list.append(alice.end_symbol)
        elif word == alice.start_word_symbol:
            char_list.append(alice.start_word_symbol)
        elif word == alice.end_word_symbol:
            char_list.append(alice.end_word_symbol)
        else:
            char_list = list(word)

        #print("char_list: ", char_list)
        return char_list


    def sample_hyperparameters(self):
        pass

    def check_customer(self):
        pass

    def print_log(self):
        print("is_char: ", self.is_char)
        print("w_hpylm.n: ", self.n)
        print(self)

    def poisson(self, lamb, k):
        return np.exp(-lamb) * ( (lamb**(k-1)) / math.factorial(k-1) )


class Gu():# n-1のcontextが渡される
    def __init__(self, context, parent):
        self.context = context
        self.parent = parent

        #print("self.parent: ", self.parent)

    def choose_add_table(self, word):
        return self.parent.choose_add_table(self.context, word)

    def choose_remove_table(self, word):
        return self.parent.choose_remove_table(self.context, word)

    def word_probability(self, word):
        return self.parent.word_probability(self.context, word)

class HPYLM():
    def __init__(self, n, c_hpylm = None, is_char = False):
        self.context_restaurant = {}
        self.n = n
        self.prior = PYP_prior(0.8, 1.0)
        self.is_char = is_char
        if n == 1:
            if is_char:
                #self.parent = G0(alice.get_num_vocabulary())
                self.parent = G0(alice.get_num_char())
            else:
                self.parent = w_G0(char_n_global, c_hpylm)
        else:
            self.parent = HPYLM(n-1, c_hpylm, is_char)

    #相席か新しいテーブルかを選ぶ
    #相席の場合はテーブル番号を返す
    def choose_add_table(self, context, word):
        if context not in self.context_restaurant:
            if self.n == 1:
                #print(context) == ()
                self.context_restaurant[context] = PYP(self.parent, self.prior) # G0から確率を引く(n=1のときparentはG0)
            else:
                #print(self.parent)
                self.context_restaurant[context] = PYP(Gu(context[1:], self.parent), self.prior)
        self.context_restaurant[context].choose_add_table(word)

    def choose_remove_table(self, context, word):
        if context in self.context_restaurant:
            self.context_restaurant[context].choose_remove_table(word)

    def word_probability(self, context, word):
        #print("word: ", word)
        if context not in self.context_restaurant:
            #if self.n == 1:
            #    return PYP(self.parent, self.prior).word_probability(word) # これをc_hpylmから引くようにする (n=1のときparentはG0) ?
            #else:
            #print("Pw: ", PYP(Gu(context[1:], self.parent), self.prior).word_probability(word))
            return PYP(Gu(context[1:], self.parent), self.prior).word_probability(word)
        return self.context_restaurant[context].word_probability(word)



#レストランのテーブルの総数と，親の客の数が等しいかどうかをチェック
    def check_customer(self):
        if self.n > 1:
            # for context, restaurant in self.context_restaurant.items():
            #     print("context: ", context)
            #     gu = restaurant.base
            #     print("gu.context: ", gu.context)

            sum_num_tables = 0
            for restaurant in self.context_restaurant.values():
                sum_num_tables += restaurant.num_tables

            sum_num_customers = 0

            for gu_restaurant in self.parent.context_restaurant.values():
                sum_num_customers += gu_restaurant.num_customers

            #print(sum_num_customers)
            #print(sum_num_tables)

            assert sum_num_tables == sum_num_customers, "expected sum_num_tables == sum_num_customers but was " + str(sum_num_tables) + "!=" + str(sum_num_customers)
            if self.n > 2:
                self.parent.check_customer()

# チェック用の関数
# レストラン内の確率が1になるかチェック
    def evaluate(self, keylist):
        for context in self.context_restaurant:
            self.context_restaurant[context].evaluate(keylist)

    def train(self, itr, ngrams_list, test_sentence): # Gibbs sampling
        perplexity_list = []
        for i in range(itr):
            for ngrams in ngrams_list:
                for ngram in ngrams:
                    if i > 0:
                        self.choose_remove_table(ngram[:-1], ngram[-1])
                    self.choose_add_table(ngram[:-1], ngram[-1])
                    #print("ngram[:-1]: ", ngram[:-1])
            perplexity_list.append(self.calc_perplexities(test_sentence))
            self.sample_hyperparameters()
            print("get_hyperparameters: ", self.get_hyperparameters())
        return perplexity_list

    def calc_perplexities(self, test_sentences):
        perplexity = 0
        log_probability = 0
        for test_sentence in test_sentences:
            for ngram in test_sentence:
                log_probability += np.log2(self.word_probability(ngram[:-1], ngram[-1]))
            log_probability = log_probability / len(test_sentence)

        print("log_probability: ", log_probability)
        perplexity = np.power(2, -log_probability)

        print("perplexity: ", perplexity)
        return perplexity

    def generate(self, max_seq_len, n):
        generate_seq = []
        for i in range(n-1):
            generate_seq.append(alice.start_symbol)

        for i in range(max_seq_len):
            rnd = np.random.uniform()
            sum_rnd = 0
            is_eos = False
            for key in alice.get_vocabulary():
                #print("context: ", tuple(generate_seq[-(n-1):]))
                #print("word: ", key)
                #print("hpylm.word_probability: ", hpylm.word_probability(tuple(generate_seq[-(n-1):]), key))
                sum_rnd += hpylm.word_probability(tuple(generate_seq[-(n-1):]), key)
                if rnd < sum_rnd:
                    if key == alice.start_symbol:
                        is_eos = True
                        break
                    else:
                        generate_seq.append(key)
                        if key == alice.end_symbol:
                            is_eos = True
                        break
            if is_eos:
                break
        return generate_seq

    def sample_hyperparameters(self):
        for restaurant in self.context_restaurant.values():
            restaurant.update_variables()

        self.prior.sample_hyperparameters()
        self.parent.sample_hyperparameters()

    def get_hyperparameters(self):
        if self.n == 1:
            return [(self.prior.d, self.prior.theta)]
        else:
            return self.parent.get_hyperparameters() + [(self.prior.d, self.prior.theta)]


        # ngrams_list: 全センテンスのngramsが入っているリスト [sentence_ngram, sentence_ngram, sentence_ngram]
        # ngrams: センテンス内のngramのタプルが入っているリスト [ngram, ngram, ngram]
        # ngram: ngramの単語列のタプル(w1,... wn)

    def print_log(self):
        print("is_char: ", self.is_char)
        print("w_hpylm.n: ", self.n)
        print(self)
        #print("self.context_restaurant: ", self.context_restaurant)

def main():
    itr = 1
    word_n = 2

    test_data, train_data = alice.get_validation_data()
    perplexity_list = []

    train_ngrams_list = []
    test_ngrams_list = []

    train_ngrams_list = alice.get_ngrams(train_data, word_n)
    test_ngrams_list = alice.get_ngrams(test_data, word_n)

    c_hpylm = HPYLM(char_n_global, is_char = True)
    w_hpylm = HPYLM(word_n, c_hpylm, is_char = False)

    perplexity_list.append(w_hpylm.train(itr, train_ngrams_list, test_ngrams_list))

    w_hpylm.print_log()
    w_hpylm.check_customer()
    #w_hpylm.evaluate(alice.get_vocabulary())

    for i in range(word_n):
        w_hpylm = w_hpylm.parent
        w_hpylm.print_log()
        #print("c_hpylm.context_restaurant: ", c_hpylm.context_restaurant)

    c_hpylm.print_log()
    c_hpylm.check_customer()
    #c_hpylm.evaluate(alice.get_chars())

    for i in range(char_n_global):
        c_hpylm = c_hpylm.parent
        c_hpylm.print_log()
        #print("c_hpylm.context_restaurant: ", c_hpylm.context_restaurant)

    # with open('test.pickle', 'wb') as f:
    #     pickle.dump(hpylm, f)

    #return perplexity_list

if __name__ == '__main__':
    main()
