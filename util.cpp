#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>

#include "main.hpp"
void printUsage(std::string& progname)
{
  std::cerr << "Usage: "<< progname <<" [TEST FILE]" << std::endl; 
}

// note: only do area optimization, ignore nets for now
void read_ckt(std::ifstream& inFile,
	      int& numModules,
	      std::vector<int>& widths,
	      std::vector<int>& heights)
{
  std::string line;
  std::getline(inFile, line);
  numModules = std::stoi(line);
  //std::vector<int> widths(numModules), heights(numModules);
  widths = std::vector<int>(numModules);
  heights = std::vector<int>(numModules);
  for (auto i = 0; i < numModules; ++i) {
    int idx;
    std::getline(inFile, line);
    std::istringstream lineStream(line);
    lineStream >> idx >> widths.at(i) >> heights.at(i);
  }
}

int visit(const int i,
	  const int numModules,
	  const int offset,
	  const std::vector<int>& graph,
	  const std::vector<int>& vWeight,
	  std::vector<int>& coord,
	  std::vector<Status>& flag)
{
  assert(flag.at(i) != Temp);
  if (flag.at(i) == Perm)
    return coord[i+offset];
  flag.at(i) = Temp;
  int longestPath = 0;
  for (auto j = 0; j < numModules; ++j) 
    if (graph[j*numModules+i] == 1)  // if there is an incoming edge, visit
      longestPath = std::max(longestPath, visit(j, numModules, offset, graph, vWeight, coord, flag) + vWeight.at(j));
  coord[i+offset] = longestPath;
  flag.at(i) = Perm;
  return longestPath;
}
