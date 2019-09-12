# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections

seq = ["A","B","A","B","C","D","A","B","A","B","C","D","C","D","A","B","A","B","A","B","C","D","C","D","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","C","D","C","D","C","D","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","C","D","C","D","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","C","D","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","C","D","C","D","B","A","B","A","B","A","C","D","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B"]

class G0(): #一様分布
    def __init__(self, V):
        self.V = V

    def choose_add_table(self, w):
        pass

    def choose_remove_table(self, w):
        pass

    def word_probability(self, w):
        return 1.0/self.V

class PYP():
    def __init__(self, base, d = 0.3, theta = 2):
        # ディスカウント係数
        self.d = d # d=0の時はCRFに等しい
        # 集中度
        self.theta = theta # theta -> ∞ で一様分布に等しくなる
        self.num_customers_eating_dish = {} #key: dish, value: 各テーブルでそのdishを食べている人数の配列
        self.table = [] # テーブルとそこに座る客の数
        self.num_customers = 0 # 全客の数
        self.num_tables = 0 # 全テーブルの数
        self.base = base #基底測度

    def add_customer(self, index, new_table, word):

        if word not in self.num_customers_eating_dish:
            self.num_customers_eating_dish[word] = []
    #新しいテーブルに座るとき
        if new_table:
            self.num_customers_eating_dish[word].append(1)
            self.table.append(1)
            self.num_customers += 1
            self.num_tables += 1

    #既存のテーブルに座るとき
        else:
            self.num_customers_eating_dish[word][index] +=1
            self.table[index] +=1
            self.num_customers += 1

    #相席か新しいテーブルかを選ぶ
    #相席の場合はテーブル番号を返す
    def choose_add_table(self, word):
        table_index = 0
        new_table = not word in self.num_customers_eating_dish
        if not new_table:
            share_prob = sum(self.num_customers_eating_dish[word]) - self.d * len(self.num_customers_eating_dish[word]) # max(0, shapre_prob)
            new_prob = (self.theta + self.d * self.num_tables) * self.base.word_probability(word)
            #一様分布から生成
            rnd = np.random.uniform(0, share_prob + new_prob)
            #既存のテーブル
            if rnd < share_prob:
                num_customer_sum = 0
                for index, num_customer in enumerate(self.num_customers_eating_dish[word]):
                    num_customer_sum += num_customer
                    if rnd < num_customer_sum:
                        table_index = index
                        break
            #新しいテーブル
            else:
                new_table = True
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

    def word_probability(self, w):
        p = 0.0
        if word in self.num_customers_eating_dish:
            p = sum(self.num_customers_eating_dish[word]) - self.d * len(self.num_customers_eating_dish[word])
        p += (self.theta + self.d * self.num_tables) * self.base.word_probability(word)
        return p


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
