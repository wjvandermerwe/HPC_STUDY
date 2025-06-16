/* this example demonstrates parallel floating point vector
  element-wise addition with a simple __global__ function.
  to compile: nvcc -I../common vector_addition.cu -o vec_add
  to run: ./vec_add
  output: prints out first few elements in the result.
*/

#include <stdlib.h>
#include <stdio.h>
//#include <sys/time.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#ifdef _WIN32
#include <windows.h>

double et_sec_h() {
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}

#else
#include <sys/time.h>
double et_sec_h() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return ((double)t.tv_sec + (double)t.tv_usec*1.e-6);
}
#endif

void initialize(float *A, int size) {
    // generate different seed for random number
//    time_t t;
//    srand((unsigned) time(&t));
    // use the same seed for random number
    srand(2025);
    for (int i=0; i<size; i++) {
        A[i] = (float)( rand() & 0xFF );
//        A[i] = (float)( rand() & 0xFF )/10.0f;
    }
}

// this kernel computes the vector sum C[i] = A[i] + B[i] on host
void vector_add_h(float *A, float *B, float *C, const size_t n) { 
    for (int i=0; i<n; i++)
        C[i] = A[i] + B[i];
}

// this kernel computes the vector sum C[i] = A[i] + B[i] on device
// each thread performs one pair-wise addition
__global__ void vector_add_d(const float *A, const float *B, float *C, const int n) {
    // compute the global element index this thread should process
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    // avoid accessing out of bounds elements
    if(i < n) {
        // sum elements
        C[i] = A[i] + B[i];
    }
}

int verify_result(float *ref, float *a, const int n) {
    double epsilon = 1.0e-6;
    bool match = 1;
    for (int i=0; i<n; i++) {
        if (abs(ref[i] - a[i]) > epsilon) {
            match = 0;
            printf("Found the first unmatch!\n");
            printf("Ref %5.2f result %5.2f at current %d\n",ref[i],a[i],i);
            break;
        }
    }
    return match; 
}

int main(void) {
    // create arrays of 1M elements
    const int num_elements = 1<<24;
    // compute the size of the arrays in bytes
    const int num_bytes = num_elements * sizeof(float);
    // for timing the sequential run
    double start_c, elapsed_c; 
    // pointers to host & device arrays
    float *device_array_a = 0;
    float *device_array_b = 0;
    float *device_array_c = 0;
    float *host_array_a   = 0;
    float *host_array_b   = 0;
    float *host_array_c   = 0;
    float *ref_array_c   = 0;

    // allocate mem for the host arrays
    host_array_a = (float*)malloc(num_bytes);
    host_array_b = (float*)malloc(num_bytes);
    host_array_c = (float*)malloc(num_bytes);
    ref_array_c = (float*)malloc(num_bytes);

    // allocate mem for the device arrays
    checkCudaErrors(cudaMalloc((void**)&device_array_a, num_bytes));
    checkCudaErrors(cudaMalloc((void**)&device_array_b, num_bytes));
    checkCudaErrors(cudaMalloc((void**)&device_array_c, num_bytes));

    // if any memory allocation failed, report an error message
    if(host_array_a == 0 || host_array_b == 0 || host_array_c == 0 || ref_array_c == 0 || \
        device_array_a == 0 || device_array_b == 0 || device_array_c == 0) {
        printf("couldn't allocate memory\n");
        return 1;
    }

    // initialize host_array_a & host_array_b
    initialize(host_array_a, num_elements); 
    initialize(host_array_b, num_elements);
    //  for(int i = 0; i < num_elements; ++i) {
    //    // make array a a linear ramp
    //    host_array_a[i] = (float)i;
    //    // make array b random
    //    host_array_b[i] = (float)rand() / RAND_MAX;
    //  }
    
    memset(ref_array_c, 0, num_bytes);
    memset(host_array_c, 0, num_bytes);
    
    // add vector at host side for result checks
    start_c = et_sec_h();
    vector_add_h(host_array_a, host_array_b, ref_array_c, num_elements);
    elapsed_c = et_sec_h() - start_c;
    
//    for(int i = 0; i<12; i++){
//        printf("A[%d] + B[%d] = C[%d] -- %6.2f + %6.2f = %6.2f\n", i, i, i, host_array_a[i], host_array_b[i], ref_array_c[i]); 
//    }
//


    // copy arrays a & b to the device memory space
    checkCudaErrors(cudaMemcpy(device_array_a,host_array_a,num_bytes,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_array_b,host_array_b,num_bytes,cudaMemcpyHostToDevice));

    // compute c = a + b on the device
//    dim3 block_size (256);
//    dim3 grid_size (num_elements / block_size.x);
    
    int block_size = 256; 
    int grid_size = num_elements / block_size;
    // deal with a possible partial final block
    if(num_elements % block_size) ++grid_size; 


    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    //  cudaEvent_t launch_begin, launch_end;
    //  checkCudaErrors(cudaEventCreate(&launch_begin));
    //  checkCudaErrors(cudaEventCreate(&launch_end));
    // record a CUDA event immediately before and after the kernel launch
    checkCudaErrors(cudaEventRecord(start,0));
    // launch the kernel
    vector_add_d<<<grid_size, block_size>>>(device_array_a, device_array_b, device_array_c, num_elements);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaEventRecord(stop,0));
    checkCudaErrors(cudaEventSynchronize(stop));
    // measure the time (ms) spent in the kernel
    float elapsed_g = 0;
    checkCudaErrors(cudaEventElapsedTime(&elapsed_g, start, stop));

    // copy the result back to the host memory space
    checkCudaErrors(cudaMemcpy(host_array_c,device_array_c,num_bytes,cudaMemcpyDeviceToHost));
    
    
//    for(int i=0; i<12; i++){
//        printf("A[%d] + B[%d] = C[%d] -- %6.2f + %6.2f = %6.2f\n", i, i, i, host_array_a[i], host_array_b[i], host_array_c[i]); 
//    }
//    
    
    if(verify_result(ref_array_c, host_array_c, num_elements)){
        printf("CPU run time: %fs\n", elapsed_c);
        printf("\nKernel run time: %fs\n\n", elapsed_g/1.0e+3);
        printf("That is a speedup: %f\n\n", (elapsed_c*1.0e+3)/(elapsed_g));
    }
    
    // deallocate memory
    free(host_array_a);
    free(host_array_b);
    free(host_array_c);
    free(ref_array_c);
    checkCudaErrors(cudaFree(device_array_a));
    checkCudaErrors(cudaFree(device_array_b));
    checkCudaErrors(cudaFree(device_array_c));
    cudaDeviceReset();
    return 0;
}
