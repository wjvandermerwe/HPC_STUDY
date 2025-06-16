
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>


#define TILE_WIDTH 8
#define BLOCK_SIZE 32
// a sequential version of matrix_multiply
void matrix_multiply_seq(float *a, float *b, float *ab, size_t width){
	int i, j, k;
	for(i=0; i<width; i++)
		for(j=0; j<width; j++){
			ab[i*width+j]=0.0;
			for(k=0; k<width; k++){
				ab[i*width+j] += a[i*width+k] * b[k*width+j];
			}
		}
}

// a simple version of matrix_multiply which issues redundant loads from off-chip global memory
__global__ void matrix_multiply_simple(float *a, float *b, float *ab, size_t width){
    // calculate the row & column index of the element
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    float result = 0;
    // do dot product between row of a and column of b
    for(int k = 0; k < width; ++k){
        result += a[row*width+k] * b[k*width+col];
    }
    // write out this thread's result
    ab[row*width+col] = result;
    }

// compare two matrix to see if they are equal -- for verification
int matrixEqual(  float *matrixA, float *matrixB, int m, int n ){
    int bad = 0;
    for ( int y = 0; y < m && !bad ; y++ )
    for ( int x = 0; x < n && !bad ; x++ ){
        if ( abs(matrixA[y*n+x] - matrixB[y*n+x]) > 1e-8 ){
            bad++;
        }
    }
    return bad;
}

int main(void){
    // create a large workload so we can easily measure the
    // performance difference of both implementations
    // note that n measures the width of the matrix, not the number of total elements
    const size_t n = 1<<8;
    //const dim3 block_size(TILE_WIDTH,TILE_WIDTH);
    const dim3 block_size(BLOCK_SIZE,BLOCK_SIZE);
    const dim3 num_blocks(n / block_size.x, n / block_size.y);

    // generate random input on the host
    float *h_a, *h_b, *h_s, *h_res;
    //std::vector<float> h_a(n*n), h_b(n*n), h_c(n*n);
    h_a = (float *)malloc(sizeof(float) * n * n);
    h_b = (float *)malloc(sizeof(float) * n * n);
    h_s = (float *)malloc(sizeof(float) * n * n);
    h_res = (float*)malloc(sizeof(float) * n * n);

    for(int i = 0; i < n*n; ++i){
    h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // allocate storage for the device
    float *d_a = 0, *d_b = 0, *d_c = 0;
    cudaMalloc((void**)&d_a, sizeof(float) * n * n);
    cudaMalloc((void**)&d_b, sizeof(float) * n * n);
    cudaMalloc((void**)&d_c, sizeof(float) * n * n);

    // copy input to the device
    cudaMemcpy(d_a, h_a, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * n * n, cudaMemcpyHostToDevice);

    // time the kernel launches using CUDA events
    cudaEvent_t launch_begin, launch_end;
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);
    //time many sequential run and take the average
    size_t num_launches = 4;
    double average_seq_time;
    struct timespec start, end;
    std::cout << "Timing sequential implementation...";
    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
        perror( "clock gettime" );
        exit( EXIT_FAILURE );
    }
    for(int i = 0; i < num_launches; i++){
        matrix_multiply_seq(h_a, h_b, h_s, n);
    }

    if( clock_gettime( CLOCK_REALTIME, &end) == -1 ) {
        perror( "clock gettime" );
        exit( EXIT_FAILURE );
    }
    //compute the time in s
    average_seq_time = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e+9;
    //take the average
    average_seq_time /= num_launches;
    std::cout << " done." << std::endl;
    std::cout << average_seq_time << "s" << std::endl;
    // launch a single "warm-up" kernel
    matrix_multiply_simple<<<num_blocks,block_size>>>(d_a, d_b, d_c, n);
    cudaMemcpy(h_res, d_c, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
    int equal = matrixEqual(h_res, h_s, n, n);
    if(equal)
        printf("Verification success.\n");
    else {
        printf("Verification failed.\n");
        num_launches = 0;
    }
    // time many kernel launches and take the average time
    float average_simple_time = 0;
    std::cout << "Timing simple implementation...";
    for(int i = 0; i < num_launches; ++i){
        // record a CUDA event immediately before and after the kernel launch
        cudaEventRecord(launch_begin,0);
        matrix_multiply_simple<<<num_blocks,block_size>>>(d_a, d_b, d_c, n);
        cudaEventRecord(launch_end,0);
        cudaEventSynchronize(launch_end);
        // measure the time spent in the kernel
        float time = 0;
        cudaEventElapsedTime(&time, launch_begin, launch_end);
        average_simple_time += time;
    }
    average_simple_time /= num_launches;
    std::cout << " done." << std::endl;
    std::cout << average_simple_time << "ms" << std::endl;

    // report the effective throughput of each kernel in GFLOPS
    // the effective throughput is measured as the number of floating point operations performed per second:
    // (one mul + one add) * N^3
    float num_ops=2 * n * n * n;
    float seq_throughput = num_ops / average_seq_time / 1000000000.0f;
    float simple_throughput = num_ops / (average_simple_time / 1000.0f) / 1000000000.0f;

    std::cout << "Matrix size: " << n << "x" << n << std::endl;
    std::cout << "Tile size: " << TILE_WIDTH << "x" << TILE_WIDTH << std::endl;
    std::cout << "Throughput of sequential implementation: " << seq_throughput << " GFLOPS" << std::endl;
    std::cout << "Throughput of simple kernel: " << simple_throughput << " GFLOPS" << std::endl;
    std::cout << "Performance improvement: simple over seqential " << simple_throughput / seq_throughput << "x" << std::endl;

    // destroy the CUDA events
    cudaEventDestroy(launch_begin);
    cudaEventDestroy(launch_end);

    // deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_s);
    free(h_res);

    return 0;
}

