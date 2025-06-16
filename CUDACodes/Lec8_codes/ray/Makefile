INC="../common"
INCFLAGS=-I$(INC)
GLUTFLAGS=-lglut -lGL
OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc

all: ray ray_noconst
ray_noconst: ray_noconst.cu
	$(NVCC) ray_noconst.cu -o ray_noconst $(INCFLAGS) $(GLUTFLAGS)
ray: ray.cu
	$(NVCC) ray.cu -o ray $(INCFLAGS) $(GLUTFLAGS)
clean:
	rm ray ray_noconst
