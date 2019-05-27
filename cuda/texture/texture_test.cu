#include <stdio.h>
// Simple transformation kernel
__global__ void transformKernel(
    float* d_output,
    cudaTextureObject_t texObj,
    int width){
    // Calculate normalized texture coordinates
    float u = threadIdx.x/(float) blockDim.x;
    // Read from texture and write to global memory
    d_output[threadIdx.x] = tex1D<float>(texObj,u);

    for (int i = 0; i < blockDim.x; i++){
        if (threadIdx.x == i){
            printf("(%.2f, %.2f )\t",u,d_output[threadIdx.x]);
        }
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 1){
        printf("\n");
    }
}

// Host code
int main(){
    int width = 10;
    float * h_data =(float *) malloc(sizeof(float)*width);;
    for (int i=0; i<width; i++){
        h_data[i]=(float) i;
    }
    int size = width*sizeof(float);

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
        32, 0, 0, 0,
        cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, size,1);
    printf("cuda malloc array\n");
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

    // Copy to device memory some data located at address h_data
    // in host memory 
    cudaMemcpyToArray(
        cuArray, 0, 0,
        h_data, size,
        cudaMemcpyHostToDevice);
    printf("cuda memcpy to array\n");
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(
        &texObj,
        &resDesc,
        &texDesc,
        NULL);
    printf("create texture \n");
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

    int num = 10;
    // Allocate result of transformation in device memory
    float* d_output;
    cudaMalloc(&d_output, num*width*sizeof(float));
    float* output = (float *) malloc(num*width*sizeof(float));

    // Invoke kernel
    transformKernel<<<1,100>>>(
        d_output,
        texObj, width);
    
    // retrieve the output
    cudaMemcpy(output,d_output,num*width*sizeof(float),cudaMemcpyDeviceToHost);

    for (int i=0; i< width; i++){
        printf("%.2f \t",h_data[i]);
    }
    printf("\n");

    // Destroy texture object
    cudaDestroyTextureObject(texObj);

    // Free device memory
    cudaFreeArray(cuArray);
    cudaFree(d_output);

    return 0;
}
