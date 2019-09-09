#include <vector>
using namespace std;

class PYP
{
public:
  PYP(int theta, double d)
  {
    m_d = d;
    m_theta = theta;
    m_n_customers = 0;
    m_n_table = 0;
  }

public:
  //ディスカウント係数
  double m_d;
  //集中度
  int m_theta;
  //全客の数
  int m_n_customers;
  //全テーブルの数
  int m_n_table;
  //テーブルとそこに座る客の数
  vector<int> m_table;

public:
  void add_customer(int index);
  void choose_table();
};
