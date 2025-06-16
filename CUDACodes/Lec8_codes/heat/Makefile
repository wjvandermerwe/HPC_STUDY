INC="../common"
INCFLAGS=-I$(INC)
GLUTFLAGS=-lglut -lGL
OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc

all: heat_1d heat_2d
heat_1d: heat_1d.cu
	$(NVCC) heat_1d.cu -o heat_1d $(INCFLAGS) $(GLUTFLAGS)
heat_2d: heat_2d.cu
	$(NVCC) heat_2d.cu -o heat_2d $(INCFLAGS) $(GLUTFLAGS)
clean:
	rm heat_1d heat_2d
