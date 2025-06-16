#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// Includes CUDA
#include <cuda_runtime.h>
// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
// simple transformation kernel
__global__ void transformKernel(float *output,cudaTextureObject_t texObj,int width,int height,float theta) {
  // calculate normalized texture coordinates
  unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;
  float u = x/(float)width;
  float v = y/(float)height;
  // transform coordinates: the following is a simple
  //geometric transform of the coordinates
  u -= 0.5f;
  v -=0.5f;
  float tu = u*cosf(theta)-v*sinf(theta)+0.5f;
  float tv = v*cosf(theta)-u*sinf(theta)+0.5f;
  // read from texture and write to global memory
  output[y*width+x] = tex2D<float>(texObj,tu,tv);
}
// host Code
int main() {
  const int height = 1024;
  const int width = 1024;
  float angle = 0.5;
  //allocate and set some host data
  float *h_data = (float*)malloc(sizeof(float)*width*height);
  for(int i=0; i<height*width; ++i){
    h_data[i]=i;
  }
  //allocate cuda array in device memory
  //the arguments: (32,0,0,0,...) specify the number of bits in each member of the
  //texture element. In the case of 1-element float, the x component is 32,
  //and the other components (y,z,w) are all 0; for example, if the type is uint2,
  //then each texture element has two members or components, in that case,
  //x and y components ned to be set to 8, respectively.
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,\
    cudaChannelFormatKindFloat);
  //declare a cuda array using keyword cudaArray_t
  cudaArray_t cuArray;
  //the memory is allocated for a cuda array on the device memory
  //using cudaMallocArray; the memory size is 1024-by-1024 floats
  checkCudaErrors(cudaMallocArray(&cuArray,&channelDesc,width,height));
  // set pitch of source, we dont have any padding
  const size_t spitch=width*sizeof(float);
  // copy data located at address h_data in host memory to device memory
  checkCudaErrors(cudaMemcpy2DToArray(cuArray,0,0,h_data,spitch,width*sizeof(float),\
    height,cudaMemcpyHostToDevice));
  // specify texture
  struct cudaResourceDesc resDesc;
  //initialize all the components of resDesc to NULL
  memset(&resDesc,0,sizeof(resDesc));
  //only set those relevant components
  //the type of resource to texture from is
  //cuda array
  resDesc.resType = cudaResourceTypeArray;
  //if the resource type is cuda array, the following
  //component must be the cuda array you declared earlier
  resDesc.res.array.array = cuArray;
  // specify texture object parameters
  struct cudaTextureDesc texDesc;
  //initialize all the components of texDesc to NULL
  memset(&texDesc,0,sizeof(texDesc));
  //only set those relevant components
  //addressing mode along x direction is wrap around
  //i.e., if you read the next element beyond the last texture element
  //(out of boundary), then the first texture element will be returned;
  //if you read the 2nd element beyond the last texture element,
  //then the 2nd texture element will be returned, and so on
  texDesc.addressMode[0] = cudaAddressModeWrap;
  //addressing mode along y direction is wrap around
  texDesc.addressMode[1] = cudaAddressModeWrap;
  //the returned value is the linear interpolation of the
  //4 (for 2D texture, then 2 for 1D etc.) texels whose
  //texture coordinates are the closest to the input texel coordinates
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  //the texture coordinates are normalized, i.e., in the range
  //of [0.0, 1.0)
  texDesc.normalizedCoords = 1;
  // with the texture resource descriptor and texture object descriptor being
  //set, it is ready to create a texture object
  cudaTextureObject_t texObj = 0;
  checkCudaErrors(cudaCreateTextureObject(&texObj,&resDesc,&texDesc,NULL));
  // allocate result of transformation in device memory
  float *output;
  //allocate device memory for output
  checkCudaErrors(cudaMalloc(&output,width*height*sizeof(float)));
  //invoke Kernel
  dim3 threadsperBlock(16,16);
  dim3 numBlocks((width+threadsperBlock.x-1)/threadsperBlock.x,\
    (height+threadsperBlock.y-1)/threadsperBlock.y);
  //texture object can be passed as an argument in kernel launch
  //but a texture reference can't be
  transformKernel<<<numBlocks, threadsperBlock>>>(output,texObj,width,height,angle);
  // copy data from device back to host
  checkCudaErrors(cudaMemcpy(h_data,output,width*height*sizeof(float),\
    cudaMemcpyDeviceToHost));
  // destroy texture object
  checkCudaErrors(cudaDestroyTextureObject(texObj));
  // free device memory
  checkCudaErrors(cudaFreeArray(cuArray));
  checkCudaErrors(cudaFree(output));
  // free host memory
  free(h_data);
  return 0;
}
