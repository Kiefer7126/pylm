# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
from pyp import PYP
from alice import Alice
import pickle

alice = Alice()

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

    def choose_add_table(self, word):
        pass

    def choose_remove_table(self, word):
        pass

    def word_probability(self, word):
        return 1.0/self.V

class Gu():# n-1のcontextが渡される
    def __init__(self, context, parent):
        self.context = context
        self.parent = parent

    def choose_add_table(self, word):
        return self.parent.choose_add_table(self.context, word)

    def choose_remove_table(self, word):
        return self.parent.choose_remove_table(self.context, word)

    def word_probability(self, word):
        return self.parent.word_probability(self.context, word)

class HPYLM():
    def __init__(self, n):
        self.context_restaurant = {}
        self.n = n
        if n == 1:
            #c = collections.Counter(seq)
            self.parent = G0(alice.get_num_vocabulary())
        else:
            self.parent = HPYLM(n-1)

    #相席か新しいテーブルかを選ぶ
    #相席の場合はテーブル番号を返す
    def choose_add_table(self, context, word):
        if context not in self.context_restaurant:
            if self.n == 1:
                self.context_restaurant[context] = PYP(self.parent)
            else:
                self.context_restaurant[context] = PYP(Gu(context[1:], self.parent))
        self.context_restaurant[context].choose_add_table(word)

    def choose_remove_table(self, context, word):
        if context in self.context_restaurant:
            self.context_restaurant[context].choose_remove_table(word)

    def word_probability(self, context, word):
        if context not in self.context_restaurant:
            if self.n == 1:
                return PYP(self.parent).word_probability(word)
            else:
                return PYP(Gu(context[1:], self.parent)).word_probability(word)
        return self.context_restaurant[context].word_probability(word)


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

def main():
    test_data, train_data = alice.get_validation_data()
    perplexity_list = []

    itr = 10
    order = 10

    for n in range(order):
        print(n+1)

        # ngrams_list: 全センテンスのngramsが入っているリスト [sentence_ngram, sentence_ngram, sentence_ngram]
        # ngrams: センテンス内のngramのタプルが入っているリスト [ngram, ngram, ngram]
        # ngram: ngramの単語列のタプル(w1,... wn)

        train_ngrams_list = []
        test_ngrams_list = []

        train_ngrams_list = alice.get_ngrams(train_data, n+1)
        test_ngrams_list = alice.get_ngrams(test_data, n+1)


        # with open('test.pickle', 'rb') as f:
        #     hpylm = pickle.load(f)

        hpylm = HPYLM(n+1)

        perplexity_list.append(hpylm.train(itr, train_ngrams_list, test_ngrams_list))

        hpylm.check_customer()
        #hpylm.evaluate(alice.get_vocabulary())

        # with open('test.pickle', 'wb') as f:
        #     pickle.dump(hpylm, f)

        #return perplexity_list

    x = np.arange(len(perplexity_list[0]))

    for n in range(9):
         plt.plot(x, np.array(perplexity_list[n]))
    plt.show()

if __name__ == '__main__':
    main()
