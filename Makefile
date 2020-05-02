NVCC        = nvcc
NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\" --relocatable-device-code true
CXX_FLAGS   = -std=c++11
CXX	    = g++

default: pso_floorplan

pso_floorplan: main.o
	$(CXX) $(CXX_FLAGS) main.o -o $@

main.o: main.cpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $^
