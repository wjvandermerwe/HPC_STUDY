# Specify the correct include path if necessary
#INC="../common"
INC="/usr/local/cuda-10.1/samples/common/inc"
INCFLAGS=-I$(INC)
GLUTFLAGS=-lglut -lGL
OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc

all: ex_tex_obj
ex_tex_obj: ex_tex_obj.cu
	$(NVCC) ex_tex_obj.cu $(INCFLAGS) -o ex_tex_obj
clean:
	rm ex_tex_obj
