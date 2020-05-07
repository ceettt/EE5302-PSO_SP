NVCC        = nvcc
NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\" --relocatable-device-code true
CXX_FLAGS   = -std=c++11
CXX	    = g++
CXX_LDFLAGS = -O3 -Wall -Wextra -lpthread
NVCC_LDFLAGS= -lcudart -lcurand -L/usr/local/cuda/lib64
NVCC_FLAGS += -O2

default: pso_floorplan_cuda

pso_floorplan: main.o util.o
	$(CXX) $(CXX_FLAGS) $(CXX_LDFLAGS) -o $@ $^

pso_floorplan_cuda: main_cuda.o util.o
	$(NVCC) $^ -o $@ $(NVCC_LDFLAGS) $(NVCC_FLAGS) $(CXX_FLAGS)

main_cuda.o: main_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $(CXX_FLAGS) -c -o $@ $< 

main.o: main.cpp util.hpp
	$(CXX) $(CXX_FLAGS) $(CXX_LDFLAGS) -c -o $@ $<

util.o: util.cpp
	$(CXX) $(CXX_FLAGS) $(CXX_LDFLAGS) -c -o $@ $<

.PHONY: clean

clean:
	rm -f *.o *~ pso_floorplan pso_floorplan_cuda
