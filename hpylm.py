# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
from pyp import PYP

seq = ["A","B","A","B","C","D","A","B","A","B","C","D","C","D","A","B","A","B","A","B","C","D","C","D","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","C","D","C","D","C","D","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","C","D","C","D","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","C","D","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","C","D","C","D","B","A","B","A","B","A","C","D","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B"]

class G0(): #一様分布
    def __init__(self, V):
        self.V = V

    def choose_add_table(self, word):
        pass

    def choose_remove_table(self, word):
        pass

    def word_probability(self, word):
        return 1.0/self.V

class Gu():
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
            c = collections.Counter(seq)
            self.parent = G0(len(c))
        else:
            self.parent = HPYLM(n-1)

    #相席か新しいテーブルかを選ぶ
    #相席の場合はテーブル番号を返す
    def choose_add_table(self, context, word):
        if context not in self.context_restaurant:
            if self.n == 1:
                self.context_restaurant[context] = PYP(self.parent)
            else:
                self.context_restaurant[context] = PYP(Gu(context, self.parent))
        self.context_restaurant[context].choose_add_table(word)

    def choose_remove_table(self, context, word):
        if context in self.context_restaurant:
            self.context_restaurant[context].choose_remove_table(word)

    def word_probability(self, context, word):
        if context not in self.context_restaurant:
            if self.n == 1:
                return PYP(self.parent).word_probability(word)
            else:
                return PYP(Gu(context, self.parent)).word_probability(word)
        return self.context_restaurant[context].word_probability(w)


def main():
    def ngram(seq, n):
        for i in range(n-1):
            seq.insert(0,"SOS")
            seq.append("EOS")
        return list(zip(*(seq[i:] for i in range(n))))

    ngrams = ngram(seq, 1)
    hpylm = HPYLM(1)

    for i in range(100000):
        for ngram in ngrams:
            if i > 0:
                hpylm.choose_remove_table(ngram[:-1], ngram[-1])
            hpylm.choose_add_table(ngram[:-1], ngram[-1])

    print(hpylm.context_restaurant[()].num_customers_eating_dish)



    #for i in range(10):
    #    hpylm.choose_add_table()

if __name__ == '__main__':
    main()
