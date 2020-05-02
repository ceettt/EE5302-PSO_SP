#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <numeric>

int main(int argc, const char *argv[])
{
  int sth1[6] = {};
  std::vector<int> sth(5);
  std::iota(std::begin<int, 6>(sth1), std::end<int, 6>(sth1), 1);
  for (int i=0; i<6; i++) {
    std::cout << sth1[i] << std::endl;
  }
}
