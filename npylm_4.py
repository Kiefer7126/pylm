# -*- coding: utf-8 -*-
"""
2019/1/16 NPYLM 単語分割メソッドの実装
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
import copy
import random

alice = Alice()
char_n_global = 3
MAX_WORD_LENGTH = 15
#test_sentence = "ALICEWASBEGINNINGTOGETVERYTIREDOFSITTINGBYHERSISTERONTHEBANK,ANDOFHAVINGNOTHINGTODOONCEORTWICESHEHADPEEPEDINTOTHEBOOKHERSISTERWASREADING,BUTITHADNOPICTURESORCONVERSATIONSINIT,ANDWHATISTHEUSEOFABOOK,THOUGHTALICEWITHOUTPICTURESORCONVERSATIONS?LASTLY,SHEPICTUREDTOHERSELFHOWTHISSAMELITTLESISTEROFHERSWOULD,INTHEAFTER-TIME,BEHERSELFAGROWNWOMANANDHOWSHEWOULDKEEP,THROUGHALLHERRIPERYEARS,THESIMPLEANDLOVINGHEARTOFHERCHILDHOODANDHOWSHEWOULDGATHERABOUTHEROTHERLITTLECHILDREN,ANDMAKETHEIREYESBRIGHTANDEAGERWITHMANYASTRANGETALE,PERHAPSEVENWITHTHEDREAMOFWONDERLANDOFLONGAGOANDHOWSHEWOULDFEELWITHALLTHEIRSIMPLESORROWS,ANDFINDAPLEASUREINALLTHEIRSIMPLEJOYS,REMEMBERINGHEROWNCHILD-LIFE,ANDTHEHAPPYSUMMERDAYS."
test_sentence = "I'MAPOORMAN,THEHATTERWENTON,ANDMOSTTHINGSTWINKLEDAFTERTHATONLYTHEMARCHHARESAIDIDIDN'T!"
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
        #return probability * self.poisson(lamb=2, k = len(char_list))
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
        #print("self.context:", self.context)
        #print("word:", word)

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

    def test_train(self, sentence):
        alpha = self.calc_forward_probability(sentence)
        word_list = self.backward_sampling(alpha, sentence)
        print(word_list)

    def train_npylm(self, itr, sentences, n):

        old_ngram_list = {}

        for i in range(itr):
            print("epoch: ", i+1)
            for sentence in random.sample(sentences, len(sentences)):
                if i == 0:
                    ngrams = self.get_ngram([sentence], n)
                    for ngram in ngrams:
                        self.choose_add_table(ngram[:-1], ngram[-1])
                if i > 0:
                    for ngram in old_ngram_list[sentence]:
                        self.choose_remove_table(ngram[:-1], ngram[-1])
                alpha = self.calc_forward_probability(sentence)
                word_list = self.backward_sampling(alpha, sentence)
                #print(word_list)
                ngrams = self.get_ngram(word_list, n)
                old_ngram_list[sentence] = ngrams
                for ngram in ngrams:
                    self.choose_add_table(ngram[:-1], ngram[-1])
            self.sample_hyperparameters()
            self.test_train(test_sentence)

        # ngrams = self.get_ngram([sentence], n)
        # #print("ngram: ", ngrams)
        # for ngram in ngrams:
        #     self.choose_add_table(ngram[:-1], ngram[-1])
        # alpha = self.calc_forward_probability(sentence)
        # word_list = self.backward_sampling(alpha, sentence)
        # for ngram in ngrams:
        #     self.choose_remove_table(ngram[:-1], ngram[-1])
        #
        # for i in range(itr):
        #     ngrams = self.get_ngram(word_list, n)
        #     for ngram in ngrams:
        #         if i > 0:
        #             self.choose_remove_table(ngram[:-1], ngram[-1])
        #         self.choose_add_table(ngram[:-1], ngram[-1])
        #     self.sample_hyperparameters()


    def get_ngram(self, sentence, n):
        seq = []
        seq = copy.copy(sentence) # ここが値参照でバグっていた

        for i in range(n - 1):
            seq.insert(0, alice.start_symbol)
        seq.append(alice.end_symbol)

        return list(zip(*(seq[i:] for i in range(n))))


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

    def calc_forward_probability(self, sentence):
        SENTENCE_LENGTH = len(sentence)
        alpha = np.zeros((SENTENCE_LENGTH+1, MAX_WORD_LENGTH+1))
        alpha[0][0] = 1
        #print("SENTENCE_LENGTH: ", SENTENCE_LENGTH)
        #print("MAX_WORD_LENGTH:", MAX_WORD_LENGTH)

        #print("sentence: ", sentence)

        for t in range(1, SENTENCE_LENGTH+1): #rangeは第一引数から第二引数のインデックスまで
            for k in range(1, MAX_WORD_LENGTH+1):
                if t == k:
                    context = (alice.start_symbol,)
                    word = sentence[0:t]
                    # print("context: ", context)
                    # print("word: ", word)
                    alpha[t][k] = self.word_probability(context = context, word = word)
                    #print("alpha[" + str(t) + "][" + str(k) + "]: " + str(alpha[t][k]))

                elif t - k > 0:
                    sum_prob = 0
                    # print("t,k: ", t,k)
                    # print("t-k+1: ", t-k+1)
                    range_index = min(t-k, MAX_WORD_LENGTH)
                    for j in range(1, range_index+1):
                        #print("prob: ", self.word_probability(context = (sentence[t-k-j:t-k],), word = sentence[t-k:t]))
                        sum_prob += self.word_probability(context = (sentence[t-k-j:t-k],), word = sentence[t-k:t]) * alpha[t-k][j]
                    alpha[t][k] = sum_prob

                    #print("alpha[" + str(t) + "][" + str(k) + "]: " + str(alpha[t][k]))
                else:
                    #print("else")
                    break

        return alpha

# 前向き確率alphaから単語長kをサンプリングする
    def backward_sampling(self, alpha, sentence):
        SENTENCE_LENGTH = len(sentence)
        #print("SENTENCE_LENGTH: ", SENTENCE_LENGTH)
        list_sampling_word = []
        t = len(sentence)
        word = alice.end_symbol

        while t > 0:
            prob_table = []
            for k in range(1, MAX_WORD_LENGTH+1):
                #print("context: ", sentence[t - k: t])
                #print("word: ", word)
                #print("prob: ", self.word_probability(context = (sentence[t - k: t],), word = word))
                #print("alpha: ", alpha[t][k])
                prob_table.append(alpha[t][k] * self.word_probability(context = (sentence[t - k: t],), word = word))
                #prob_table.append(alpha[t][k])

            #print("prob_table: ", prob_table)

            sampling_k = self.sampling_from_distribution(prob_table)
            #print("sampling_k: ", sampling_k)
            #print("sentence: ", sentence)
            word = sentence[t - sampling_k:t]
            #print("word: ", word)
            list_sampling_word.append(word)
            t -= sampling_k
            #print("t: ", t)

        return list_sampling_word[::-1] #逆順

    def sampling_from_distribution(self, prob_distribution):
        rnd = np.random.rand() * sum(prob_distribution)
        cumulative_prob = 0
        sampling_k = 1
        #print("prob_distribution: ", prob_distribution)

        for index, prob in enumerate(prob_distribution):
            cumulative_prob += prob
            #print("index: ", index)
            #print("prob:" , prob)
            if rnd < cumulative_prob:
                sampling_k = index+1

                return sampling_k


def main():
    itr = 5
    word_n = 2

    test_data, train_data = alice.get_validation_data()
    perplexity_list = []

    train_ngrams_list = []
    test_ngrams_list = []

    train_ngrams_list = alice.get_ngrams(train_data, word_n)
    test_ngrams_list = alice.get_ngrams(test_data, word_n)

    # with open('test.pickle', 'rb') as f:
    #     w_hpylm = pickle.load(f)
    # print("w_hpylm.context_restaurant: ", w_hpylm.context_restaurant)

    c_hpylm = HPYLM(char_n_global, is_char = True)
    w_hpylm = HPYLM(word_n, c_hpylm, is_char = False)

    #perplexity_list.append(w_hpylm.train(itr, sentence, word_n))
    #sentences = ["今日はいい天気ですね", "明日はいい天気になるといいですね", "これはテストです", "明日の天気は何ですか"]
    #sentences = ["ILIKEMUSICILIKEMUSIC", "ILOVEMUSICILOVEMUSIC", "ISINGASONGISINGASONG","NOMUSICNOLIFENOMUSICNOLIFE"]
    #sentences = ["WHENSHETHOUGHTITOVERAFTERWARDS,ITOCCURREDTOHERTHATSHEOUGHTTOHAVEWONDEREDATTHIS,BUTATTHETIMEITALLSEEMEDQUITENATURAL","BUTWHENTHERABBITACTUALLYTOOKAWATCHOUTOFITSWAISTCOAT-POCKET,ANDLOOKEDATIT,ANDTHENHURRIEDON,ALICESTARTEDTOHERFEET,FORITFLASHEDACROSSHERMINDTHATSHEHADNEVERBEFORESEENARABBITWITHEITHERAWAISTCOAT-POCKET,ORAWATCHTOTAKEOUTOFIT,ANDBURNINGWITHCURIOSITY,SHERANACROSSTHEFIELDAFTERIT,ANDFORTUNATELYWASJUSTINTIMETOSEEITPOPDOWNALARGERABBIT-HOLEUNDERTHEHEDGE","INANOTHERMOMENTDOWNWENTALICEAFTERIT,NEVERONCECONSIDERINGHOWINTHEWORLDSHEWASTOGETOUTAGAIN","EITHERTHEWELLWASVERYDEEP,ORSHEFELLVERYSLOWLY,FORSHEHADPLENTYOFTIMEASSHEWENTDOWNTOLOOKABOUTHERANDTOWONDERWHATWASGOINGTOHAPPENNEXT","FIRST,SHETRIEDTOLOOKDOWNANDMAKEOUTWHATSHEWASCOMINGTO,BUTITWASTOODARKTOSEEANYTHING","THENSHELOOKEDATTHESIDESOFTHEWELL,ANDNOTICEDTHATTHEYWEREFILLEDWITHCUPBOARDSANDBOOK-SHELVES"]
    #sentences = ["12345678910111213141516171819202122232425262728293031323334353637383940"]
    #sentences = ["THELONGGRASSRUSTLEDATHERFEETASTHEWHITERABBITHURRIEDBYTHEFRIGHTENEDMOUSESPLASHEDHISWAYTHROUGHTHENEIGHBOURINGPOOLSHECOULDHEARTHERATTLEOFTHETEACUPSASTHEMARCHHAREANDHISFRIENDSSHAREDTHEIRNEVER-ENDINGMEAL,ANDTHESHRILLVOICEOFTHEQUEENORDERINGOFFHERUNFORTUNATEGUESTSTOEXECUTIONONCEMORETHEPIG-BABYWASSNEEZINGONTHEDUCHESS'SKNEE,WHILEPLATESANDDISHESCRASHEDAROUNDITONCEMORETHESHRIEKOFTHEGRYPHON,THESQUEAKINGOFTHELIZARD'SSLATE-PENCIL,ANDTHECHOKINGOFTHESUPPRESSEDGUINEA-PIGS,FILLEDTHEAIR,MIXEDUPWITHTHEDISTANTSOBSOFTHEMISERABLEMOCKTURTLE."]

    sentences = alice.get_sequences()
    w_hpylm.train_npylm(itr, sentences, word_n)

    w_hpylm.print_log()
    w_hpylm.check_customer()
    #w_hpylm.evaluate(alice.get_vocabulary())
    print("w_hpylm.context_restaurant: ", w_hpylm.context_restaurant)

    with open('w_hpylm_epoch'+ str(itr) +'.pickle', 'wb') as f:
        pickle.dump(w_hpylm, f)

    with open('c_hpylm_epoch'+ str(itr) +'.pickle', 'wb') as f:
        pickle.dump(c_hpylm, f)

    for i in range(word_n):
        w_hpylm = w_hpylm.parent
        w_hpylm.print_log()
        #print("w_hpylm.context_restaurant: ", w_hpylm.context_restaurant)

    c_hpylm.print_log()
    c_hpylm.check_customer()
    #c_hpylm.evaluate(alice.get_chars())
    print("c_hpylm.context_restaurant: ", c_hpylm.context_restaurant)

    for i in range(char_n_global):
        c_hpylm = c_hpylm.parent
        c_hpylm.print_log()
        #print("c_hpylm.context_restaurant: ", c_hpylm.context_restaurant)



    #return perplexity_list

if __name__ == '__main__':
    main()
