#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <string>

#include "util.hpp"
int main(int argc, const char *argv[])
{
  // parameter parsing
  std::vector<std::string> args(argv, argv+argc);

  try {
    std::ifstream ckt_file(args.at(1));
    if (!ckt_file.is_open()) {
      std::cerr << "Cannot open file:\t" << args.at(1) << std::endl;
      exit(1);
    }
    read_ckt(ckt_file);
    ckt_file.close();
  } catch (const std::out_of_range& e) {
    std::cerr << "Not enough parameters." << std::endl;
    printUsage(args.at(0));
    exit(1);
  }

  return 0;
}
