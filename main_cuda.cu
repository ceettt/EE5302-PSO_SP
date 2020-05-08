#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <iterator>
#include <memory>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>

#include "util.hpp"

#define NDEBUG

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__constant__ int dSizes[1024]; // constant memory storing the dimensions

constexpr int numParticles = 32;

constexpr float cSwap = 0.6f; // probability

// deleter functor
template <typename T>
struct cuda_deleter {
  cuda_deleter() {}
  void operator()(T* ptr) {
    cudaFree(ptr);
  }
};


__host__ __device__
int paddingTo32(int num) {
  return (num+31)/32*32;
}

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, decltype(cuda_deleter<T>())>;

// build graph in the memory according to the position relationship
__device__
void buildGraph(int& tIdx,
		int* HCG,
		int* VCG,
		int* OrderP, // __shared__
		int* OrderN, // __shared__
		int& rIncomeHCG, // register
		int& rIncomeVCG, // register
		const int numModules)
{
  rIncomeHCG = 0;
  rIncomeVCG = 0;
  for (auto j = 0; j < numModules; ++j) {
    int idx = numModules*paddingTo32(numModules)*(blockIdx.x) + j*paddingTo32(numModules) + tIdx; // edge j -> tIdx
    HCG[idx] = 0;
    VCG[idx] = 0;
    if (j != tIdx) {
      if (OrderN[j] < OrderN[tIdx]) {
	if (OrderP[j] < OrderP[tIdx]) {// left of
	  HCG[idx] = 1;
	  rIncomeHCG += 1;
	}
	else {// below
	  VCG[idx] = 1;
	  rIncomeVCG += 1;
	}
      }
    }
    __syncthreads();
  }
}

// topological visit graph, see report for detailed algorithm
// Note: Expect Qstart and Qend be 0
__device__
void topovisit(int& tIdx,
	       int& pIdx,
	       int* Graph,
	       int& Income, // count of income edge to vertex tIdx
	       bool& isAddedToQueue,
	       int* sCoord, // __shared__
	       int* Queue, // __shared__
	       int* path, // __shared__
	       int* pQstart, // ptr to __shared__
	       int* pQend, // ptr to __shared__
	       int* pDim, // ptr to __shared__
	       const int dOffset, // if width 0 height 512
	       const int numModules)
{
  //__syncthreads();
  // initialize Queue
  isAddedToQueue = false;
  sCoord[tIdx] = 0;
  Queue[tIdx] = -1;
  int idxQ = -1;
  __syncthreads();
  // if there is no incoming edge and it is not added to queue do that
  if ((Income == 0) && (!isAddedToQueue)) { 
    idxQ = atomicAdd(pQend, 1); // get an index in Q
    Queue[idxQ] = tIdx;
    isAddedToQueue = true;
  }
  __syncthreads();
  int Qstart = 0, Qend = 0;
  Qend = *pQend;
  __syncthreads();
  while (Qstart < Qend) { // there is element in Queue
    int currVertex = Queue[Qstart]; // a vertex with no incoming edge from edge not visited
#ifndef NDEBUG
    if (currVertex == -1) {
      atomicAdd(pQstart, 1);
      //printf("Q %d %d %d\n",pIdx, Qstart, Qend);
      //if (isAddedToQueue != false) {
      //printf("%d %d\n", tIdx, idxQ);
      //}
      //if (tIdx == 0) {
      //for (auto i = 0; i < *pQend; ++i) {
	  //printf("Q %d %d\n", i, Queue[i]);
      //}
	//}
      //__threadfence();
      //asm("trap;");
    }
    
    __syncthreads();

    if (*pQstart != 0) {
      printf("%d %d %d %d %d %d %d %d\n", pIdx, tIdx, currVertex, Qend, Qstart, Income, (int)isAddedToQueue, idxQ);
      __syncthreads();
      asm("trap;");
    }
#endif
    ++Qstart;
    __syncthreads();

    // set outgoing edge as visited by subtracting the count to income[tIdx]
    int idx = numModules*paddingTo32(numModules)*(blockIdx.x)
      + currVertex*paddingTo32(numModules) + tIdx; // edge currVertex -> tIdx
    if (Graph[idx] == 1) 
      Income -= 1;
    
    // get longest past from all predecents of currVertex
    path[tIdx] = 0;
    __syncthreads();
    idx = numModules*paddingTo32(numModules)*(blockIdx.x)
      + tIdx*paddingTo32(numModules) + currVertex; // edge tIdx -> currVertex
    if (Graph[idx] == 1) {
      path[tIdx] = sCoord[tIdx] + dSizes[tIdx+dOffset];
    }
    __syncthreads();

    // reduction to get max 
    for (auto i = numModules/2; i>0; i>>=1) {
      if ((tIdx <= i) && (tIdx+i < numModules)) 
	path[tIdx] = max(path[tIdx], path[tIdx+i]);
      __syncthreads();
    }
    // fix odd problem that last cycle is not performed
    if (tIdx == 0) 
      path[0] = max(path[0], path[1]);
      __syncthreads();
    // update Queue
    if ((Income == 0) && (!isAddedToQueue)) {
      int idxQ = atomicAdd(pQend, 1); // get an index in Q
      Queue[idxQ] = tIdx;
      isAddedToQueue = true;
    }
    __syncthreads();
    Qend = *pQend;
    __syncthreads();
    // increment, store result
    if (tIdx == 0) {
      *pQstart = 0;
      sCoord[currVertex] = path[0];
      *pDim = max(*pDim, path[0] + dSizes[currVertex+dOffset]);
    }
    __syncthreads();
  }
}

__global__
void initialization(int* lBestGammaP,
		    int* lBestGammaN,
		    int* GammaP,
		    int* GammaN,
		    int* lBestArea,
		    int* lastArea,
		    int* HCG,
		    int* VCG,
		    int* wC,
		    int* hC,
		    unsigned long long seed,
		    curandState* states,
		    const int numModules)
{
  int tIdx = threadIdx.x;
  int offset = blockIdx.x*paddingTo32(numModules);
  int pIdx = blockIdx.x;
  
  int chipArea=0;

  int rIncomeHCG=0, rIncomeVCG=0; // income count for this vertex
  bool isAddedToQueue=false;
  
  __shared__ int sGammaP[512];
  __shared__ int sGammaN[512];
  __shared__ int sOrderP[512];
  __shared__ int sOrderN[512];
  __shared__ int sWidCoord[512];
  __shared__ int sHeiCoord[512];
  __shared__ int queue[512];
  __shared__ int path[512];
  __shared__ int chipWidth, chipHeight;
  __shared__ int Qstart;
  __shared__ int Qend;

  curand_init(seed, tIdx+offset, 0, &states[tIdx+offset]);
  sOrderP[tIdx] = -1;
  sOrderN[tIdx] = -1;
  sGammaP[tIdx] = tIdx;
  sGammaN[tIdx] = tIdx;

  // random swap variables to initialize
  int stride = 1 << int(ceilf(log2f((float)numModules))-1);
  
  __syncthreads();
  for (; stride >=1; stride >>=1) { //
    if ((!((tIdx / stride) & 1)) && (tIdx+stride < numModules)) {// if it is a valid swap
      if (curand(&states[tIdx+offset]) & 1) {
	int temp = sGammaP[tIdx];
	sGammaP[tIdx] = sGammaP[tIdx+stride];
	sGammaP[tIdx+stride] = temp;
      }
      if (curand(&states[tIdx+stride]) & 1) {
	int temp = sGammaN[tIdx];
	sGammaN[tIdx] = sGammaN[tIdx+stride];
	sGammaN[tIdx+stride] = temp;
      }
    }
    __syncthreads();
  }
  
  sOrderP[sGammaP[tIdx]] = tIdx;
  sOrderN[sGammaN[tIdx]] = tIdx;
  __syncthreads();
  buildGraph(tIdx,
	     HCG,
	     VCG,
	     sOrderP,
	     sOrderN,
	     rIncomeHCG,
	     rIncomeVCG,
	     numModules);
  __syncthreads();

  if (tIdx == 0) {
    Qstart = 0;
    Qend = 0;
    chipWidth = 0;
    chipHeight = 0;
  }
  __syncthreads();
  topovisit(tIdx,
	    pIdx,
	    HCG,
	    rIncomeHCG,
	    isAddedToQueue,
	    sWidCoord, // __shared__
	    queue, // __shared__
	    path, // __shared__
	    &Qstart, // ptr to __shared__
	    &Qend, // ptr to __shared__
	    &chipWidth, // ptr to __shared__
	    0, // if width 0 height 512
	    numModules);

  if (tIdx == 0) {
    if (Qend != numModules)
      printf("Unexpected end at %d\n", Qend);
    Qstart = 0;
    Qend = 0;
  }
  __syncthreads();
  
  topovisit(tIdx,
	    pIdx,
	    VCG,
	    rIncomeVCG,
	    isAddedToQueue,
	    sHeiCoord, // __shared__
	    queue, // __shared__
	    path, // __shared__
	    &Qstart, // ptr to __shared__
	    &Qend, // ptr to __shared__
	    &chipHeight, // ptr to __shared__
	    512, // if width 0 height 512
	    numModules);
  
  if (tIdx == 0) {
    if (Qend != numModules)
      printf("Unexpected end at %d\n", Qend);
  }
  
  __syncthreads();
  wC[tIdx+offset] = sWidCoord[tIdx];
  hC[tIdx+offset] = sHeiCoord[tIdx];
  lBestGammaP[tIdx+offset] = sGammaP[tIdx];
  lBestGammaN[tIdx+offset] = sGammaN[tIdx];
  GammaP[tIdx+offset] = sGammaP[tIdx];
  GammaN[tIdx+offset] = sGammaN[tIdx];
  if (tIdx == 0) {
    chipArea = chipWidth*chipHeight;
    lastArea[pIdx] = chipArea;
    lBestArea[pIdx] = chipArea;
  }
}

// function used to perform random swap to gamma sequence
__device__
void swapSequence(int& tIdx,
		  int sourceOffset,
		  int targetOffset, // globalbest ones should be 0 here
		  int* swapTo, // __share__ var
		  int* sSource, // __shared__ array
		  int* sTarget, // __shared__ array
		  int* sGamma, // __shared, this is the swap taking place
		  int* Gamma, // global, only read
		  int* TargetGamma, // global
		  curandState* states,
		  const int numModules)
{
  sSource[tIdx] = Gamma[tIdx+sourceOffset];
  sTarget[tIdx] = TargetGamma[tIdx+targetOffset];
  __syncthreads();
  for (auto i=0; i<numModules; ++i) {
    if (sTarget[i] == sSource[tIdx])
      *swapTo = tIdx;
    __syncthreads();
    if (tIdx == 0) // let thread 0 swap
      if (i != tIdx) { // only swap if the id is different
	int temp = sSource[*swapTo];
	sSource[*swapTo] = sSource[i];
	sSource[i] = temp;
	if (curand_uniform(&states[sourceOffset]) < cSwap) {
	  temp = sGamma[*swapTo];
	  sGamma[*swapTo] = sGamma[i];
	  sGamma[i] = temp;
	}
      }
    __syncthreads();
  }
}

__global__
void update(int* GammaP,
	    int* GammaN,
	    int* lBestGammaP,
	    int* lBestGammaN,
	    int* gBestGammaP,
	    int* gBestGammaN,
	    int* HCG,
	    int* VCG,
	    int* wC,
	    int* hC,
	    int* lBestArea,
	    int* lastArea,
	    curandState* states,
	    const int numModules)
{
  int tIdx = threadIdx.x;
  int offset = blockIdx.x*paddingTo32(numModules);
  int pIdx = blockIdx.x;
  
  int chipArea=0;

  int rIncomeHCG=0, rIncomeVCG=0; // income count for this vertex
  bool isAddedToQueue=false;
  int cSwapRandom = ceilf(0.01*numModules);
  
  __shared__ int sGammaP[512];
  __shared__ int sGammaN[512];
  __shared__ int sSource[512];
  __shared__ int sTarget[512];
  __shared__ int sOrderP[512];
  __shared__ int sOrderN[512];
  __shared__ int sWidCoord[512];
  __shared__ int sHeiCoord[512];
  __shared__ int queue[512];
  __shared__ int path[512];
  __shared__ int chipWidth, chipHeight;
  __shared__ int swapTo;
  __shared__ int Qstart;
  __shared__ int Qend;

  // update velocity and search space

  sGammaP[tIdx] = GammaP[tIdx+offset];
  sGammaN[tIdx] = GammaN[tIdx+offset];
  
  // local Positive swap
  swapSequence(tIdx,
	       offset,
	       offset,
	       &swapTo,
	       sSource, // __shared__
	       sTarget, // __shared__
	       sGammaP, // __shared, this is the swap taking place
	       GammaP, // global mem, only read
	       lBestGammaP, // global mem
	       states,
	       numModules);
  __syncthreads();

  // local Negative swap
  swapSequence(tIdx,
	       offset,
	       offset,
	       &swapTo,
	       sSource, // __shared__
	       sTarget, // __shared__
	       sGammaN, // __shared, this is the swap taking place
	       GammaN, // global mem, only read
	       lBestGammaN, // global mem
	       states,
	       numModules);
  __syncthreads();

  // global Positive swap
  swapSequence(tIdx,
	       offset,
	       0,
	       &swapTo,
	       sSource, // __shared__
	       sTarget, // __shared__
	       sGammaP, // __shared, this is the swap taking place
	       GammaP, // global mem, only read
	       gBestGammaP, // global mem
	       states,
	       numModules);
  __syncthreads();

  // global Negative swap
  swapSequence(tIdx,
	       offset,
	       0,
	       &swapTo,
	       sSource, // __shared__
	       sTarget, // __shared__
	       sGammaN, // __shared, this is the swap taking place
	       GammaN, // global mem, only read
	       gBestGammaN, // global mem
	       states,
	       numModules);
  __syncthreads();

  // add some random swap
  if (tIdx == 0) 
    for (auto i=0; i < cSwapRandom; ++i) {
      unsigned int j = curand(&states[tIdx+offset]) % numModules;
      unsigned int k = curand(&states[tIdx+offset]) % numModules;
      unsigned int type = curand(&states[tIdx+offset]) % 3;
      int temp = 0;
      if (type != 1) {
	temp = sGammaP[j];
	sGammaP[j] = sGammaP[k];
	sGammaP[k] = temp;
      }
      if (type != 2) {
	temp = sGammaN[j];
	sGammaN[j] = sGammaN[k];
	sGammaN[k] = temp;
      }
    }

  
  // build Sequence Pair

  __syncthreads();
  sOrderP[tIdx] = -1;
  sOrderN[tIdx] = -1;
  __syncthreads();
  GammaP[tIdx+offset] = sGammaP[tIdx];
  GammaN[tIdx+offset] = sGammaN[tIdx];
  sOrderP[sGammaP[tIdx]] = tIdx;
  sOrderN[sGammaN[tIdx]] = tIdx;
  __syncthreads();
  if ((sOrderP[tIdx] == -1) || (sOrderN[tIdx] == -1)) {
    __threadfence();
    asm("trap;");
  }
  
  __syncthreads();

  buildGraph(tIdx,
	     HCG,
	     VCG,
	     sOrderP,
	     sOrderN,
	     rIncomeHCG,
	     rIncomeVCG,
	     numModules);
  __syncthreads();

  if (tIdx == 0) {
    Qstart = 0;
    Qend = 0;
    chipWidth = 0;
    chipHeight = 0;
  }
  __syncthreads();
  topovisit(tIdx,
	    pIdx,
	    HCG,
	    rIncomeHCG,
	    isAddedToQueue,
	    sWidCoord, // __shared__
	    queue, // __shared__
	    path, // __shared__
	    &Qstart, // ptr to __shared__
	    &Qend, // ptr to __shared__
	    &chipWidth, // ptr to __shared__
	    0, // if width 0 height 512
	    numModules);

  if (tIdx == 0) {
    
    if (Qend != numModules) {
      printf("1Unexpected end at %d, %d\n", Qend, Qstart);
    }
    Qstart = 0;
    Qend = 0;
  }
  __syncthreads();
  
  topovisit(tIdx,
	    pIdx,
	    VCG,
	    rIncomeVCG,
	    isAddedToQueue,
	    sHeiCoord, // __shared__
	    queue, // __shared__
	    path, // __shared__
	    &Qstart, // ptr to __shared__
	    &Qend, // ptr to __shared__
	    &chipHeight, // ptr to __shared__
	    512, // if width 0 height 512
	    numModules);
  
  if (tIdx == 0) {
    if (Qend != numModules) {
      printf("2Unexpected end at %d, %d\n", Qend, Qstart);
    }
  }
  
  __syncthreads();
  chipArea = chipWidth*chipHeight;
  wC[tIdx+offset] = sWidCoord[tIdx];
  hC[tIdx+offset] = sHeiCoord[tIdx];
  
  // update local best
  if (chipArea < lBestArea[pIdx]) {
    if (tIdx == 0) 
      lBestArea[pIdx] = chipArea;
    lBestGammaP[tIdx+offset] = sGammaP[tIdx];
    lBestGammaN[tIdx+offset] = sGammaN[tIdx];
  }
  if (tIdx == 0) 
    lastArea[pIdx] = chipArea;
}

__global__
void copyGlobalBest(int pIdx,
		    int* gBestGammaP,
		    int* gBestGammaN,
		    int* GammaP,
		    int* GammaN,
		    int* gBestWidC,
		    int* gBestHeiC,
		    int* wC,
		    int* hC,
		    const int numModules)
{
  int tIdx = threadIdx.x;
  int offset = pIdx*paddingTo32(numModules);
  gBestGammaP[tIdx] = GammaP[tIdx+offset];
  gBestGammaN[tIdx] = GammaN[tIdx+offset];
  gBestWidC[tIdx] = wC[tIdx+offset];
  gBestHeiC[tIdx] = hC[tIdx+offset];
}

int main(int argc, const char *argv[])
{
  // Timing
  using Time = std::chrono::high_resolution_clock;
  using us = std::chrono::microseconds;
  using fsec = std::chrono::duration<float>;


  // parameter parsing
  std::vector<std::string> args(argv, argv+argc);
  int numModules;
  std::vector<int> widths, heights;
  std::vector<int> sizes(1024, 0);
  try {
    std::ifstream ckt_file(args.at(1));
    if (!ckt_file.is_open()) {
      std::cerr << "Cannot open file:\t" << args.at(1) << std::endl;
      exit(1);
    }
    read_ckt(ckt_file, numModules, widths, heights);
    ckt_file.close();
    std::copy(std::begin(widths), std::end(widths), std::begin(sizes));
    std::copy(std::begin(heights), std::end(heights), std::begin(sizes)+512);
  } catch (const std::out_of_range& e) {
    std::cerr << "Not enough parameters." << std::endl;
    printUsage(args.at(0));
    exit(1);
  }
  // Allocate memory needed
  auto myCudaMalloc = [](size_t size)
    {
     void* ptr;
     cudaMalloc(&ptr, size);
     return ptr;
    };
  auto myCudaMallocManaged = [](size_t size)
    {
     void* ptr;
     cudaMallocManaged(&ptr, size);
     return ptr;
    };
  cudaSetDevice(0);
  cudaFree(0);
  std::ofstream result_file("result.txt");
  auto start = Time::now();
  // global scope
  cuda_unique_ptr<int> gBestGammaP((int*)myCudaMallocManaged(numModules*sizeof(int)));
  cuda_unique_ptr<int> gBestGammaN((int*)myCudaMallocManaged(numModules*sizeof(int)));
  cuda_unique_ptr<int> gBestWidC((int*)myCudaMallocManaged(numModules*sizeof(int)));
  cuda_unique_ptr<int> gBestHeiC((int*)myCudaMallocManaged(numModules*sizeof(int)));
  int gBestArea = -1;
  // padded memories, local to block
  int padded = paddingTo32(numModules);
  cuda_unique_ptr<int> lBestGammaP((int*)myCudaMalloc(padded*numParticles*sizeof(int)));
  cuda_unique_ptr<int> lBestGammaN((int*)myCudaMalloc(padded*numParticles*sizeof(int)));
  cuda_unique_ptr<int> GammaP((int*)myCudaMalloc(padded*numParticles*sizeof(int)));
  cuda_unique_ptr<int> GammaN((int*)myCudaMalloc(padded*numParticles*sizeof(int)));
  cuda_unique_ptr<int> wC((int*)myCudaMalloc(padded*numParticles*sizeof(int)));
  cuda_unique_ptr<int> hC((int*)myCudaMalloc(padded*numParticles*sizeof(int)));
  cuda_unique_ptr<curandState> states((curandState*)myCudaMalloc(padded*numParticles*sizeof(curandState)));
  // storage for HCG and VCG
  cuda_unique_ptr<int> HCG((int*)myCudaMalloc(numModules*padded*numParticles*sizeof(int)));
  cuda_unique_ptr<int> VCG((int*)myCudaMalloc(numModules*padded*numParticles*sizeof(int)));
  // local to block, each block maintain one
  cuda_unique_ptr<int> lastArea((int*)myCudaMallocManaged(numParticles*sizeof(int)));
  cuda_unique_ptr<int> lBestArea((int*)myCudaMalloc(numParticles*sizeof(int)));


  cudaMemcpyToSymbol(dSizes, sizes.data(), 1024*sizeof(int));
  
  
  
  // initialize velocity and search space with random variable
  std::random_device rd;

  initialization<<<numParticles, numModules>>>(lBestGammaP.get(),
					       lBestGammaN.get(),
					       GammaP.get(),
					       GammaN.get(),
					       lBestArea.get(),
					       lastArea.get(),
					       HCG.get(),
					       VCG.get(),
					       wC.get(),
					       hC.get(),
					       rd(),
					       states.get(),
					       numModules);
  
  cudaDeviceSynchronize();
  gpuErrchk( cudaPeekAtLastError() );
  int minParIdx = std::distance(lastArea.get(),
				std::min_element(lastArea.get(), lastArea.get()+numParticles));

  gBestArea = lastArea.get()[minParIdx];
  copyGlobalBest<<<1, numModules>>>(minParIdx,
				    gBestGammaP.get(),
				    gBestGammaN.get(),
				    GammaP.get(),
				    GammaN.get(),
				    gBestWidC.get(),
				    gBestHeiC.get(),
				    wC.get(),
				    hC.get(),
				    numModules);   
  //for (auto i = 0; i < numParticles; ++i) 
  //std::cout << lastArea.get()[i] << "\t";
  std::cout << std::endl;
  cudaDeviceSynchronize();
  std::cout << gBestArea << std::endl;

  //  std::cout << "Best at this round appears at pIdx:" << minParIdx
  //	    << "\tWith Area:" << gBestArea << std::endl;
  int counter = 0;
  int cycle = 0;
  while (counter < 100) {
    update<<<numParticles, numModules>>>(GammaP.get(),
					 GammaN.get(),
					 lBestGammaP.get(),
					 lBestGammaN.get(),
					 gBestGammaP.get(),
					 gBestGammaN.get(),
					 HCG.get(),
					 VCG.get(),
					 wC.get(),
					 hC.get(),
					 lBestArea.get(),
					 lastArea.get(),
					 states.get(),
					 numModules);
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );
    minParIdx = std::distance(lastArea.get(),
			      std::min_element(lastArea.get(), lastArea.get()+numParticles));    
    ++counter;
    if (gBestArea > lastArea.get()[minParIdx]) {
      counter = 0;
      gBestArea = lastArea.get()[minParIdx];
      copyGlobalBest<<<1, numModules>>>(minParIdx,
					gBestGammaP.get(),
					gBestGammaN.get(),
					GammaP.get(),
					GammaN.get(),
					gBestWidC.get(),
					gBestHeiC.get(),
					wC.get(),
					hC.get(),
					numModules);
      std::cout << cycle << "\t" << gBestArea << std::endl; 
      cudaDeviceSynchronize();
    }
    //for (auto i = 0; i < numParticles; ++i) 
    // std::cout << lastArea.get()[i] << "\t";
    //std::cout << std::endl;  
    
    ++cycle;
  }
  std::cout << "Best Area:" << gBestArea  << "\tUsing "<< cycle << " Cycles"<< std::endl;
  //for (auto i = 0; i < numModules; ++i) 
  //std::cout << i << "\t" << gBestGammaP.get()[i] << "\t" << gBestGammaN.get()[i] << std::endl;
  // Timing
  auto stop = Time::now();
  fsec fs = stop - start;
  us d = std::chrono::duration_cast<us>(fs);
  std::cout << "Program took \t" << fs.count() << "s" << std::endl
	    << "\tor \t" << d.count() << "us" << std::endl;
  // Note: TURN OFF UNIFIED MEMORY PROFILE
  std::vector<int>
    gBestGammaP_h(numModules),
    gBestGammaN_h(numModules),
    gBestWidC_h(numModules),
    gBestHeiC_h(numModules);
  std::copy(gBestGammaP.get(), gBestGammaP.get()+numModules, std::begin(gBestGammaP_h));
  std::copy(gBestGammaN.get(), gBestGammaN.get()+numModules, std::begin(gBestGammaN_h));
  std::copy(gBestWidC.get(), gBestWidC.get()+numModules, std::begin(gBestWidC_h));
  std::copy(gBestHeiC.get(), gBestHeiC.get()+numModules, std::begin(gBestHeiC_h));
  write_ckt(result_file, gBestArea, numModules, gBestGammaP_h, gBestGammaN_h, gBestWidC_h, gBestHeiC_h);
  return 0;
}
