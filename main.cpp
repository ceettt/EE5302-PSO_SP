#include <iostream>
#include <iterator>
#include <fstream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <string>
#include <random>
#include <thread>
#include <chrono>
#include <cassert>

#include "util.hpp"
#include "main.hpp"
constexpr int numParticles = 64;
int numThreads = std::thread::hardware_concurrency();
int numPPerT = numParticles / numThreads;

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}

int main(int argc, const char *argv[])
{
  // Timing
  using Time = std::chrono::high_resolution_clock;
  using us = std::chrono::microseconds;
  using fsec = std::chrono::duration<float>;

  auto start = Time::now();
  // parameter parsing
  std::vector<std::string> args(argv, argv+argc);
  int numModules;
  std::vector<int> widths, heights;
  try {
    std::ifstream ckt_file(args.at(1));
    if (!ckt_file.is_open()) {
      std::cerr << "Cannot open file:\t" << args.at(1) << std::endl;
      exit(1);
    }
    read_ckt(ckt_file, numModules, widths, heights);
    /*
    for (auto i = 0; i < numModules; ++i) {
      std::cout << "Idx:" << i << "\twidth:" << widths.at(i)
		<< "\theight:" << heights.at(i) << std::endl; 
		}*/
    ckt_file.close();
  } catch (const std::out_of_range& e) {
    std::cerr << "Not enough parameters." << std::endl;
    printUsage(args.at(0));
    exit(1);
  }
  // particle global variables
  std::vector<int> gBestGammaP(numModules), gBestGammaN(numModules);
  std::vector<int> lBestGammaP(numParticles*numModules), lBestGammaN(numParticles*numModules);
  std::vector<int> GammaP(numParticles*numModules), GammaN(numParticles*numModules);
  std::vector<int> wC(numParticles*numModules), hC(numParticles*numModules);
  std::vector<int> lastArea(numParticles);
  std::vector<int> lBestArea(numParticles);
  std::vector<std::thread> threads(numThreads);
  int gBestArea = -1;
  std::vector<int> gBestWidC(numModules), gBestHeiC(numModules);
  // following should be done with every particle
  // shuffle to random vector
  std::cout << numThreads << std::endl;
  for (auto tIdx=0; tIdx < numThreads; ++tIdx) {
    threads.at(tIdx) = std::thread
      ([&, tIdx]{
	 // initialization
	 std::random_device rd;
	 std::mt19937 gen(rd());

	 std::vector<int> particles(numPPerT);
	 std::iota(std::begin(particles), std::end(particles), tIdx*numPPerT);
	 //std::cout << "Thread ID:" << tIdx
	 //	   << "\tExecuting Particle " << particles
	 //	   << std::endl;
	 for (const auto pIdx: particles) {
	   // shuffle initial sequential pair
	   int oBegin = pIdx*numModules;
	   int oEnd = oBegin + numModules;
	   
	   std::iota(std::begin(GammaP)+oBegin, std::begin(GammaP)+oEnd, 0);
	   std::iota(std::begin(GammaN)+oBegin, std::begin(GammaN)+oEnd, 0);
	   std::shuffle(std::begin(GammaP)+oBegin, std::begin(GammaP)+oEnd, gen);
	   std::shuffle(std::begin(GammaN)+oBegin, std::begin(GammaN)+oEnd, gen);

	   // construct HCG and VCG
  
	   // assign order for each
	   // note order can be in shared memory
	   std::vector<int> OrderP(numModules), OrderN(numModules);
	   int sizeMatrix = numModules * numModules;
	   std::vector<int> HCG(sizeMatrix, 0), VCG(sizeMatrix, 0);
	   for (auto i = 0; i < numModules; ++i) {
	     OrderP.at(GammaP.at(i+oBegin)) = i;
	     OrderN.at(GammaN.at(i+oBegin)) = i;
	   }

	   // note: in cuda HCG and VCG must be preallocated
	   for (auto i = 0; i < numModules; ++i) {
	     for (auto j = 0; j < numModules; ++j) {
	       if (i != j) {
		 if (OrderN.at(i) < OrderN.at(j)) {
		   if (OrderP.at(i) < OrderP.at(j))  // left of
		     HCG[i*numModules+j]=1;
		   else // below
		     VCG[i*numModules+j]=1;
		 }
	       }
	     }
	   }
  
	   // Toposort and get area
	   // Note: flag can be in shared memory
	   std::vector<Status> flags(numModules, None);
	   int wChip = 0, hChip = 0;
	   // toposort HCG
	   for (auto i = 0; i < numModules; ++i) {
	     wChip = std::max(wChip, widths.at(i)+visit(i, numModules, oBegin, HCG, widths, wC, flags));
	   }
	   std::fill(std::begin(flags), std::end(flags), None);
	   for (auto i = 0; i < numModules; ++i) {
	     hChip = std::max(hChip, heights.at(i)+visit(i, numModules, oBegin, VCG, heights, hC, flags));
	   }
	   //std::cout << "Particle Id:" << pIdx <<"\tTotal Area:" << wChip*hChip << std::endl;
	   lBestArea[pIdx] = wChip*hChip;
	   lastArea[pIdx] = wChip*hChip;
	   std::copy(std::begin(GammaP)+oBegin, std::begin(GammaP)+oEnd, std::begin(lBestGammaP)+oBegin);
	   std::copy(std::begin(GammaN)+oBegin, std::begin(GammaN)+oEnd, std::begin(lBestGammaN)+oBegin);
	 }
       });
  }
  for (auto& t: threads)
    t.join();
  //assert(lBestGammaP == GammaP);
  //assert(lBestGammaN == GammaN);
  int minParIdx = std::distance(std::begin(lastArea),
				std::min_element(std::begin(lastArea), std::end(lastArea)));
  gBestArea = lastArea[minParIdx];
  int minOffBegin = minParIdx*numModules;
  int minOffEnd = (minParIdx+1)*numModules;
  
  std::copy(std::begin(GammaP)+minOffBegin, std::begin(GammaP)+minOffEnd, std::begin(gBestGammaP));
  std::copy(std::begin(GammaN)+minOffBegin, std::begin(GammaN)+minOffEnd, std::begin(gBestGammaN));
  std::copy(std::begin(wC)+minOffBegin, std::begin(wC)+minOffEnd, std::begin(gBestWidC));
  std::copy(std::begin(hC)+minOffBegin, std::begin(hC)+minOffEnd, std::begin(gBestHeiC));

  int counter = 0; // stop if for 10 consecutive cycle area is not improving
  int cycle = 0;
  
  int cSwapLocal = int(0.4*numModules);
  int cSwapGlobal = int(0.4*numModules);
  int cSwapKeep = numModules - cSwapLocal - cSwapGlobal;

  // global variable for switch operators
  std::vector<int>
    sAP(numModules*numParticles, 0),
    sBP(numModules*numParticles, 0),
    sAN(numModules*numParticles, 0),
    sBN(numModules*numParticles, 0);
  
  while (counter < 100) {
    threads = std::vector<std::thread>(numThreads);
    for (auto tIdx=0; tIdx < numThreads; ++tIdx) {
      threads.at(tIdx) = std::thread
	([&, tIdx]{
	   // initialization
	   std::vector<int> particles(numPPerT);
	   std::iota(std::begin(particles), std::end(particles), tIdx*numPPerT);
	   for (const auto pIdx: particles) {
	     int oBegin = pIdx * numModules;
	     int oEnd = oBegin + numModules;
  
	     // copy first cSwapKeep Ops to the end of queue;
	     std::copy(std::begin(sAP)+oBegin, std::begin(sAP)+oBegin+cSwapKeep, std::begin(sAP)+oEnd-cSwapKeep);
	     std::copy(std::begin(sAN)+oBegin, std::begin(sAN)+oBegin+cSwapKeep, std::begin(sAN)+oEnd-cSwapKeep);
	     std::copy(std::begin(sBP)+oBegin, std::begin(sBP)+oBegin+cSwapKeep, std::begin(sBP)+oEnd-cSwapKeep);
	     std::copy(std::begin(sBN)+oBegin, std::begin(sBN)+oBegin+cSwapKeep, std::begin(sBN)+oEnd-cSwapKeep);
  
	     std::vector<int> SourceP(std::begin(GammaP)+oBegin, std::begin(GammaP)+oEnd);
	     std::vector<int> SourceN(std::begin(GammaN)+oBegin, std::begin(GammaN)+oEnd);
	     std::vector<int> TargetP(std::begin(lBestGammaP)+oBegin, std::begin(lBestGammaP)+oEnd);
	     std::vector<int> TargetN(std::begin(lBestGammaN)+oBegin, std::begin(lBestGammaN)+oEnd);
	     int offset = 0;
	     // calculate swap for local
	     for (auto i = 0; i < cSwapLocal; ++i) {
	       int j = std::find(std::begin(SourceP)+i, std::end(SourceP), TargetP[i]) - std::begin(SourceP);
	       sAP[oBegin+offset+i] = i;
	       sBP[oBegin+offset+i] = j;
	       int temp = SourceP[i];
	       SourceP[i] = SourceP[j];
	       SourceP[j] = temp;
	       j = std::find(std::begin(SourceN)+i, std::end(SourceN), TargetN[i]) - std::begin(SourceN);
	       sAN[oBegin+offset+i] = i;
	       sBN[oBegin+offset+i] = j;
	       temp = SourceN[i];
	       SourceN[i] = SourceN[j];
	       SourceN[j] = temp;
	     }

	     SourceP = std::vector<int>(std::begin(GammaP)+oBegin, std::begin(GammaP)+oEnd);
	     SourceN = std::vector<int>(std::begin(GammaN)+oBegin, std::begin(GammaN)+oEnd);
	     TargetP = std::vector<int>(gBestGammaP);
	     TargetN = std::vector<int>(gBestGammaN);
	     offset = cSwapLocal;
	     // calculate swap for global
	     for (auto i = 0; i < cSwapGlobal; ++i) {
	       int j = std::find(std::begin(SourceP)+i, std::end(SourceP), TargetP[i]) - std::begin(SourceP);
	       sAP[oBegin+offset+i] = i;
	       sBP[oBegin+offset+i] = j;
	       int temp = SourceP[i];
	       SourceP[i] = SourceP[j];
	       SourceP[j] = temp;
	       j = std::find(std::begin(SourceN)+i, std::end(SourceN), TargetN[i]) - std::begin(SourceN);
	       sAN[oBegin+offset+i] = i;
	       sBN[oBegin+offset+i] = j;
	       temp = SourceN[i];
	       SourceN[i] = SourceN[j];
	       SourceN[j] = temp;
	     }
  
	     // swap Gamma PN
	     for (auto i = 0; i < numModules; ++i) {
	       int temp = GammaP[oBegin + sAP[oBegin+i]];
	       GammaP[oBegin + sAP[oBegin+i]] = GammaP[oBegin + sBP[oBegin+i]];
	       GammaP[oBegin + sBP[oBegin+i]] = temp;
	       temp = GammaN[oBegin + sAN[oBegin+i]];
	       GammaN[oBegin + sAN[oBegin+i]] = GammaN[oBegin + sBN[oBegin+i]];
	       GammaN[oBegin + sBN[oBegin+i]] = temp;
	     }

	     // construct HCG and VCG
  
	     // assign order for each
	     // note order can be in shared memory
	     std::vector<int> OrderP(numModules), OrderN(numModules);
	     int sizeMatrix = numModules * numModules;
	     std::vector<int> HCG(sizeMatrix, 0), VCG(sizeMatrix, 0);
	     for (auto i = 0; i < numModules; ++i) {
	       OrderP.at(GammaP.at(i+oBegin)) = i;
	       OrderN.at(GammaN.at(i+oBegin)) = i;
	     }

	     // note: in cuda HCG and VCG must be preallocated
	     for (auto i = 0; i < numModules; ++i) {
	       for (auto j = 0; j < numModules; ++j) {
		 if (i != j) {
		   if (OrderN.at(i) < OrderN.at(j)) {
		     if (OrderP.at(i) < OrderP.at(j))  // left of
		       HCG[i*numModules+j]=1;
		     else // below
		       VCG[i*numModules+j]=1;
		   }
		 }
	       }
	     }
  
	     // Toposort and get area
	     // Note: flag can be in shared memory
	     std::vector<Status> flags(numModules, None);
	     int wChip = 0, hChip = 0;
	     // toposort HCG
	     for (auto i = 0; i < numModules; ++i) {
	       wChip = std::max(wChip, widths.at(i)+visit(i, numModules, oBegin, HCG, widths, wC, flags));
	     }
	     std::fill(std::begin(flags), std::end(flags), None);
	     for (auto i = 0; i < numModules; ++i) {
	       hChip = std::max(hChip, heights.at(i)+visit(i, numModules, oBegin, VCG, heights, hC, flags));
	     }
	     
	     // update local best
	     lastArea[pIdx] = wChip*hChip;
	     if (lBestArea[pIdx] > lastArea[pIdx]) {
	       lBestArea[pIdx] = lastArea[pIdx];
	       std::copy(std::begin(GammaP)+oBegin, std::begin(GammaP)+oEnd, std::begin(lBestGammaP)+oBegin);
	       std::copy(std::begin(GammaN)+oBegin, std::begin(GammaN)+oEnd, std::begin(lBestGammaN)+oBegin);
	     }
	   }
	 });
    }
    for (auto& t: threads) {
      t.join();
    }
    // update global best
    minParIdx = std::distance(std::begin(lastArea),
				  std::min_element(std::begin(lastArea), std::end(lastArea)));
    std::cout << "\t" << lastArea[minParIdx] << std::endl;
    if (gBestArea > lastArea[minParIdx]) {
      gBestArea = lastArea[minParIdx];
      minOffBegin = minParIdx*numModules;
      minOffEnd = (minParIdx+1)*numModules;
  
      std::copy(std::begin(GammaP)+minOffBegin, std::begin(GammaP)+minOffEnd, std::begin(gBestGammaP));
      std::copy(std::begin(GammaN)+minOffBegin, std::begin(GammaN)+minOffEnd, std::begin(gBestGammaN));
      std::copy(std::begin(wC)+minOffBegin, std::begin(wC)+minOffEnd, std::begin(gBestWidC));
      std::copy(std::begin(hC)+minOffBegin, std::begin(hC)+minOffEnd, std::begin(gBestHeiC));
    }
    
    /*
    std::cout << GammaP << std::endl
	      << gBestGammaP << std::endl
	      << GammaN << std::endl
	      << gBestGammaN << std::endl;
    */
    std::cout << gBestArea << std::endl;
    ++counter;
  }
  

  
  std::cout << "Best Area:" << gBestArea << std::endl;
  // Timing
  auto stop = Time::now();
  fsec fs = stop - start;
  us d = std::chrono::duration_cast<us>(fs);
  std::cout << "Program took \t" << fs.count() << "s" << std::endl
	    << "\tor \t" << d.count() << "us" << std::endl;

  return 0;
}
