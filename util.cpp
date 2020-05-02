#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

void printUsage(std::string& progname)
{
  std::cerr << "Usage: "<< progname <<" [TEST FILE]" << std::endl; 
}

// note: only do area optimization, ignore nets for now
void read_ckt(std::ifstream& inFile)
{
  std::string line;
  std::getline(inFile, line);
  int numModules = std::stoi(line);
  std::vector<int> widths(numModules), heights(numModules);
  for (auto i = 0; i < numModules; ++i) {
    int idx;
    std::getline(inFile, line);
    std::istringstream lineStream(line);
    lineStream >> idx >> widths.at(i) >> heights.at(i);
  }
  for (auto i = 0; i < numModules; ++i) {
    std::cout << "Module Idx: " << i
	      << " Module width:" << widths.at(i)
	      << " Module height:" << heights.at(i)
	      << std::endl;
  }
}
