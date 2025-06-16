INC="../Common"
INCFLAGS=-I$(INC)
GLUTFLAGS=-lglut -lGL
OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc

all: matrix_multiplication histo_template
matrix_multiplication: matrix_multiplication.cu
	$(NVCC) $(INCFLAGS) matrix_multiplication.cu -o matrix_multiplication
matrix_transpose: matrix_transpose.cu
	$(NVCC) $(INCFLAGS) matrix_transpose.cu -o matrix_transpose
histo_template: histo_template.cu
	$(NVCC) $(INCFLAGS) histo_template.cu -o histo_template
clean:
	rm matrix_multiplication histo_template
