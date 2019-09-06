# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ディスカウント係数
d = 0.4 # d=0の時はCRFに等しい

# 集中度
theta = 2 # theta -> ∞ で一様分布に等しくなる
table = []
num_customers = 0
num_table = 0

def main():
    #table.append(1)
    #print(table[0])
    for i in range(1000):
        choose_table()
    print(table)
    print(num_table)
    np_table = np.array(table)
    plt.bar(range(0,len(table)), np.sort(np_table)[::-1])
    plt.show()
    #plt.savefig("a.png")


def add_customer(index):
    global num_customers, table, num_table

#新しいテーブルに座るとき
    if index == -1:
        table.append(1)
        num_customers += 1

#既存のテーブルに座るとき
    else:
        table[index] +=1
        num_customers += 1
    num_table = len(table)


#相席か新しいテーブルかを選ぶ
#相席の場合はテーブル番号を返す
def choose_table():
    global num_customers, theta, d, num_table
    share_prob = num_customers - d
    new_prob = theta + d * num_table

    rnd = np.random.uniform(0, share_prob + new_prob)
    #相席
    if rnd < share_prob:
        num_customer_sum = 0
        for index, num_customer in enumerate(table):
            num_customer_sum += num_customer
            if rnd < num_customer_sum:
                add_customer(index)
                break;

    #新しいテーブル
    else:
        add_customer(-1)


if __name__ == '__main__':
    main()
