// CUDA Runtime
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

typedef unsigned int uint;
typedef unsigned char uchar;
#define BIN_NO 256

__global__ void transpose(const float *A, float *B, int size){
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            B[j*size + i] = A[i*size + j];
        }
    }
}

__global__ void g_tree_reduction(float *A,  int size){
    int t = threadIdx.x;

    for (size_t stride = 1; stride < blockDim.x; stride *= 2)
    {
        if((stride * 2) % t == 0)
            A[t] += A[t+stride];
    }
}

__global__ void sharedReduction(const float *A, float *sum, int size){
    extern __shared__ float buffer[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    buffer[tid] = (idx < size) ? A[idx]:0;
    __syncthreads();

    for(int i = blockDim.x/2; i > 0; i >>= 1){
        if(tid < i)
            buffer[tid] += A[tid + i];
        __syncthreads();    
    }

    if (tid == 0) 
        sum[blockIdx.x] = buffer[0];
}


int main(int argc, char **argv)
{



    // for (int exp = 9; exp < 12; exp++){
    //     int N = 1 << exp;
    //     size_t bytes = N * (size_t)N * sizeof(float);
    //     float *d_A, *d_B;
    //     cudaMalloc(&d_A, bytes); // on host
    //     cudaMalloc(&d_B, bytes);
    //     cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    //     dim3 block(16,16);
    //     dim3 grid((N+15)/16, (N+15)/16);
    //     transpose<<<grid, block>>>(d_A, d_B, N);
    // }


}
