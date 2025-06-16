/*
  This example is using cudaMalloc and cudaMemcpy
  to transfer data between host and device explicitly.
  To compile: nvcc -I../common double.cu -o double
  To run: ./double

  upon run, the following code outputs:
    All elements were doubled? FALSE
  find the error(s) in the code and get output as
    All elements were doubled? TRUE
*/
#include <stdio.h>
#include <helper_cuda.h>

void init(int *a, int N) {
  int i;
  for (i = 0; i < N; ++i) {
    a[i] = i;
  }
}

__global__ void doubleElements(int *a, int N) {
  int i;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= N) {
    a[i] *= 2;
  }
}

bool checkElementsAreDoubled(int *a, int N) {
  int i;
  for (i = 0; i < N; ++i) {
    if (a[i] != i*2)
    {
       printf("a[%d] = %d, should be %d\n", i, a[i], i*2);
       return false;
    }
  }
  return true;
}

int main() {
  int N = 10000;
  //int *a;
  int *a, *host_a;

  size_t size = N * sizeof(int);

  //checkCudaErrors(cudaMallocManaged(&a, size));
  //init(a, N);

  host_a = (int *)malloc(size);
  checkCudaErrors(cudaMalloc((void**)&a, size));
  init(host_a, N);
  checkCudaErrors(cudaMemcpy(a, host_a, size, cudaMemcpyHostToDevice));


  //the size of this grid is 256*32 = 8192.
  size_t threads_per_block = 250;
  size_t number_of_blocks = 40;
  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  getLastCudaError("doubleElements execution failed\n");
  cudaDeviceSynchronize();
  checkCudaErrors(cudaMemcpy(host_a, a, size, cudaMemcpyDeviceToHost));
  bool areDoubled = checkElementsAreDoubled(host_a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");
  free(host_a);
  checkCudaErrors(cudaFree(a));
  cudaDeviceReset();
  return 0;
}
