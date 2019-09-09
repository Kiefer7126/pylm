#include <iostream>
#include <ostream>
#include <iterator>
#include <random>
#include <algorithm>
#include "pyp.h"
using namespace std;

  void PYP::add_customer(int index)
  {
    //新しいテーブルに座るとき
    if(index == -1){m_table.push_back(1);}
    //既存のテーブルに座るとき
    else{m_table[index] += 1;}

    m_n_customers += 1;
    m_n_table = m_table.size();
  }

  void PYP::choose_table()
  {
    double share_prob = m_n_customers - m_d;
    double new_prob = m_theta + m_d * m_n_table;

    std::random_device rnd;     // 非決定的な乱数生成器を生成
    std::mt19937 mt(rnd());     //  メルセンヌ・ツイスタの32ビット版、引数は初期シード値
    std::uniform_int_distribution<> rand100(0, share_prob + new_prob);
    double arrow = rand100(mt);
    if(arrow < share_prob)
    {
      int n_customers_sum = 0;
      for(int i = 0; i < m_table.size(); i++)
      {
        n_customers_sum += m_table[i];
        if(arrow < n_customers_sum)
        {
          PYP::add_customer(i);
          break;
        }
      }
    }else{PYP::add_customer(-1);}
  }

int main(){
  PYP restaurant(2, 0.5);
  for(int i = 0; i < 1000; i++)
  {
    restaurant.choose_table();
  }

  std::sort(restaurant.m_table.begin(),restaurant.m_table.end(),std::greater<int>());
  for(int i = 0; i<restaurant.m_table.size(); i++)
  {
    cout << restaurant.m_table[i] << ",";
  }

//  std::copy(restaurant.m_table.begin(), restaurant.m_table.end(),
  //std::ostream_iterator<int>(cout));


  return 0;
}
