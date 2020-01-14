# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections

#seq = ["A","B","A","B","C","D","A","B","A","B","C","D","C","D","A","B","A","B","A","B","C","D","C","D","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","C","D","C","D","C","D","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","C","D","C","D","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","C","D","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","C","D","C","D","B","A","B","A","B","A","C","D","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B"]

class PYP_prior():

    def __init__(self, d, theta):

        # ディスカウント
        self.d = d
        # ディスカウントの推定のベータ分布のパラメタ a, b
        self.a = 1.0
        self.b = 1.0

        # 集中度の推定のガンマ分布のパラメタ α, β
        self.theta = theta
        self.alpha = 1.0
        self.beta = 1.0

    def sample_hyperparameters(self):
        self.d = np.random.beta(self.a, self.b)

        self.theta = np.random.gamma(self.alpha, self.beta)
        if self.theta <= 0:
            self.theta = 10.0
        self.theta = np.log10(self.theta)


class G0(): #一様分布
    def __init__(self, V):
        self.V = V
        self.num_customers = V

    def choose_add_table(self, w):
        pass

    def choose_remove_table(self, w):
        pass

    def word_probability(self, w):
        return 1.0/self.V

class PYP():
    # ディスカウント係数
    # self.d = d # d=0の時はCRFに等しい
    # self.theta = theta # theta -> ∞ で一様分布に等しくなる

    def __init__(self, base, prior):
        self.num_customers_eating_dish = {} #key: dish, value: 各テーブルでそのdishを食べている人数の配列
        self.table = [] # テーブルとそこに座る客の数
        self.num_customers = 0 # 全客の数
        self.num_tables = 0 # 全テーブルの数
        self.base = base # 基底測度
        self.prior = prior

    def add_customer(self, index, new_table, word):
        if word not in self.num_customers_eating_dish:
            self.num_customers_eating_dish[word] = []
    #新しいテーブルに座るとき
        if new_table:
            self.num_customers_eating_dish[word].append(1)
            #self.num_customers_eating_dish[word] += [1] <- ここあやしいけど上と同じ意味っぽい

            self.table.append(1)
            #self.num_customers += 1
            self.num_tables += 1
            self.base.choose_add_table(word)

    #既存のテーブルに座るとき
        else:
            self.num_customers_eating_dish[word][index] +=1
            self.table[index] +=1
            #self.num_customers += 1
        self.num_customers +=1

    #相席か新しいテーブルかを選ぶ
    def choose_add_table(self, word):
        table_index = 0
        new_table = not word in self.num_customers_eating_dish
        if not new_table:
            share_prob = sum(self.num_customers_eating_dish[word]) - self.prior.d * len(self.num_customers_eating_dish[word]) # max(0, shapre_prob)
            new_prob = (self.prior.theta + self.prior.d * self.num_tables) * self.base.word_probability(word)
            #一様分布から生成
            rnd = np.random.uniform(0, share_prob + new_prob)
            #新しいテーブル
            if rnd < new_prob:
                new_table = True
            #既存のテーブル
            else:
                num_customer_sum = new_prob
                for index, num_customer in enumerate(self.num_customers_eating_dish[word]):
                    #num_customer_sum += num_customer
                    if rnd < num_customer_sum + num_customer - self.prior.d:
                        table_index = index
                        break
                    num_customer_sum += num_customer - self.prior.d

        self.add_customer(table_index, new_table, word)

    def choose_remove_table(self, word):
        rnd = np.random.randint(0, sum(self.num_customers_eating_dish[word]))
        table_index = 0
        sum_num_customer = 0
        for index, num_customer in enumerate(self.num_customers_eating_dish[word]):
            sum_num_customer += num_customer
            if rnd < sum_num_customer:
                table_index = index
                break

        self.remove_customer(table_index, word)


    def remove_customer(self, index, word):
        self.num_customers_eating_dish[word][index] -= 1
        self.num_customers -= 1

        # table index の word を食べている客がいないとき
        if not self.num_customers_eating_dish[word][index]:
            del self.num_customers_eating_dish[word][index]
            self.num_tables -= 1
            self.base.choose_remove_table(word)

        if not len(self.num_customers_eating_dish[word]):
            del self.num_customers_eating_dish[word]

    def word_probability(self, word):
        p = 0.0
        #print("self.base: ", self.base)
        if word in self.num_customers_eating_dish:
            p = sum(self.num_customers_eating_dish[word]) - self.prior.d * len(self.num_customers_eating_dish[word])
        p += (self.prior.theta + self.prior.d * self.num_tables) * self.base.word_probability(word)
        return p / (self.prior.theta + self.num_customers)

# 評価用の関数
# レストラン内の確率が1になる
    def evaluate(self, keylist):
        sum_probability = 0
        #print("keylist: ", keylist)
        for word in keylist:
            #print(word)
            #print(self.word_probability(word))
            sum_probability += self.word_probability(word)
        #print("num_customers_eating_dish: ", self.num_customers_eating_dish)
        print("sum_probability: ", sum_probability)
        #assert (1 - sum_probability) < 0.001, "expected sum_probability is 1 but was " + str(sum_probability)

    def update_variables(self):
        if self.num_tables >= 2:
            xu = np.random.beta(self.prior.theta + 1, self.num_customers - 1)
            self.prior.beta -= np.log10(xu)

            for i in range(1, self.num_tables):
                yui = np.random.binomial(1, self.prior.theta / (self.prior.theta + self.prior.d * i))
                self.prior.a += (1 - yui)
                self.prior.alpha += yui

        for w in self.num_customers_eating_dish:
            for cuwk in self.num_customers_eating_dish[w]:
                if cuwk >= 2:
                    for j in range(1, cuwk):
                        zuwkj = np.random.binomial(1, (j - 1) / (j - self.prior.d))
                        self.prior.b += (1 - zuwkj)

def main():
    c = collections.Counter(seq)
    V = len(c)
    print(V)
    base = G0(V)
    restaurant = PYP(base)
    for word in seq:
        restaurant.choose_add_table(word)

        def print_log():
            print("num_tables: ", restaurant.num_tables)
            print("num_customers_eating_dish[A]: ", restaurant.num_customers_eating_dish["A"])
            print("num_customers_eating_dish[B]: ", restaurant.num_customers_eating_dish["B"])
            print("sum_num_customers_eating_dish[A]: ", sum(restaurant.num_customers_eating_dish["A"]))
            print("sum_num_customers_eating_dish[B]: ", sum(restaurant.num_customers_eating_dish["B"]))


    np_table = np.array(restaurant.table)
    plt.bar(range(0,restaurant.num_tables), np.sort(np_table)[::-1])
    plt.show()
    print_log()
    #plt.savefig("a.png")

    #for word in seq:
    #    restaurant.choose_remove_table(word)

    c = collections.Counter(seq)
    print("len(c): ", len(c))
    keylist = list(c.keys())
    print("keylist: ", keylist)

    # Gbbs sampling
    for i in range(100000):
        rnd = np.random.randint(0, len(c))
        #print("rnd: ", rnd)
        restaurant.choose_remove_table(keylist[rnd])
        restaurant.choose_add_table(keylist[rnd])

    print_log()


if __name__ == '__main__':
    main()
