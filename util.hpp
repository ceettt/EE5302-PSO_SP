#ifndef UTIL_HPP
#define UTIL_HPP
#include "main.hpp"

void printUsage(std::string& progname);
void read_ckt(std::ifstream& inFile,
	      int& numModules,
	      std::vector<int>& widths,
	      std::vector<int>& heights);
int visit(const int i,
	  const int numModules,
	  const int offset,
	  const std::vector<int>& graph,
	  const std::vector<int>& vWeight,
	  std::vector<int>& coord,
	  std::vector<Status>& flag);

#endif