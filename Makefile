NVCC        = nvcc
NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\" --relocatable-device-code true
CXX_FLAGS   = -std=c++11 -Wall -Wextra -O2 -lpthread
CXX	    = g++

default: pso_floorplan

pso_floorplan: main.o util.o
	$(CXX) $(CXX_FLAGS) -o $@ $^

main.o: main.cpp util.hpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

util.o: util.cpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

.PHONY: clean

clean:
	rm -f *.o *~ pso_floorplan
