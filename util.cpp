#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <iterator>

#include "main.hpp"
void printUsage(std::string& progname)
{
  std::cerr << "Usage: "<< progname <<" [TEST FILE]" << std::endl; 
}

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out.seekp(-1, std::ios_base::cur);
    out.seekp(-1, std::ios_base::cur);
    out << "] ";
  }
  return out;
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

void write_ckt(std::ofstream& outFile,
	       int area,
	       int numModules,
	       std::vector<int>& GammaP,
	       std::vector<int>& GammaN,
	       std::vector<int>& widC,
	       std::vector<int>& heiC)
{
  outFile << "Best Area:" << area << std::endl;
  outFile << "Given by following Sequence Pair" << std::endl;
  outFile << "Positive Sequence:" << GammaP << std::endl;
  outFile << "Negative Sequence:" << GammaN << std::endl;
  outFile << "Following is the resulting coordinate" << std::endl; 
  outFile << "label\t\tX\t\tY" << std::endl;
  for (auto i=0; i < numModules; ++i) {
    outFile << i << "\t\t"
	    << widC.at(i) << "\t\t"
	    << heiC.at(i) << std::endl;
  }
}
