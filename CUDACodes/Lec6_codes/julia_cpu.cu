/*to compile: nvcc -I../common julia_cpu.cu -o julia_cpu -lglut -lGL
  to tun: ./julia_cpu*/

//#include "../common/book.h"
#include <cpu_bitmap.h>
#include <cstdio>
#include <omp.h>
#define DIM 512

struct cuComplex {
    float   r;
    float   i;
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    float magnitude2( void ) { return r * r + i * i; }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int julia( int x, int y ) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-1.0, 0.1);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernel(unsigned char *ptr) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    int offset = x + y * gridDim.x;

    int juliaValue = julia( x, y );

    ptr[offset*4 + 0] = 255 * juliaValue;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}

void save_ppm(const char* filename, unsigned char* data) {
    FILE* f = fopen(filename, "wb");
    fprintf(f, "P6\n%d %d\n255\n", DIM, DIM);       // header
    for (int i = 0; i < DIM * DIM; ++i)             // write RGB, skip A
        fwrite(data + i * 4, 1, 3, f);
    fclose(f);
}

int main( void ) {
    CPUBitmap bitmap( DIM, DIM );
    unsigned char *ptr;
    cudaCheckError( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );

    dim3 grid(DIM, DIM);
    kernel<<<grid, 1>>>( dev_bitmap );

    cudaCheckError( cudaMemcpyw( bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost ) );

    // save_ppm("julia.ppm", ptr);

}
