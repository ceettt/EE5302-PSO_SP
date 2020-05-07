# EE5302 Final Project: GPU Accelerated Floor-planning Using Particle Swarm Optimization

Member: Hanzhao Yu (yuxx0839), Tonglin Chen (chen5202)

Branches:
 - master: velocity based on random applying swap operators
 - oldvel: velocity based on truncation of swap operators, no cuda implementation
 - vel1: velocity in continuous space

Compiling options:
```
make pso_floorplan	# compile cpu version, require c++11
make pso_floorplan_cuda # compile gpu version, should be working on ECE GPU labs
```

Usage:
```
./[program name] [TEST FILE]
```
Several testbench files are included in `./test/`
Note that all the optimization are based on area only

License:
The code is solely used for this project. All right reserved.
