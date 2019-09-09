# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

class PYP():
    def __init__(self, d = 0.3, theta = 2):
        # ディスカウント係数
        self.d = d # d=0の時はCRFに等しい
        # 集中度
        self.theta = theta # theta -> ∞ で一様分布に等しくなる
        self.table = [] # テーブルとそこに座る客の数
        self.num_customers = 0 # 全客の数
        self.num_table = 0 # 全テーブルの数

    def add_customer(self, index):
    #新しいテーブルに座るとき
        if index == -1:
            self.table.append(1)
            self.num_customers += 1

    #既存のテーブルに座るとき
        else:
            self.table[index] +=1
            self.num_customers += 1
        self.num_table = len(self.table)

    #相席か新しいテーブルかを選ぶ
    #相席の場合はテーブル番号を返す
    def choose_table(self):
        share_prob = self.num_customers - self.d #max(0, shapre_prob)
        new_prob = self.theta + self.d * self.num_table
        #一様分布から生成
        rnd = np.random.uniform(0, share_prob + new_prob)
        #相席
        if rnd < share_prob:
            num_customer_sum = 0
            for index, num_customer in enumerate(self.table):
                num_customer_sum += num_customer
                if rnd < num_customer_sum:
                    self.add_customer(index)
                    break;
        #新しいテーブル
        else:
            self.add_customer(-1)

def main():
    restaurant = PYP()
    for i in range(1000):
        restaurant.choose_table()
    print(restaurant.table)
    print(restaurant.num_table)
    np_table = np.array(restaurant.table)
    plt.bar(range(0,restaurant.num_table), np.sort(np_table)[::-1])
    plt.show()
    #plt.savefig("a.png")

if __name__ == '__main__':
    main()
