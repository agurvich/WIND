#include <stdio.h>

texture<float, 1, cudaReadModeElementType> tex;

__device__ float index_texture(float index,int texture_size){
    printf("%d, %.2f, %d\t",threadIdx.x,index,texture_size);
    return 0.5 + index*float(texture_size);
}
__global__ void kernel(int M, int N, float *d_out){

    for (int i = 0; i < blockDim.x; i++){
        if (threadIdx.x == i){
            float v = index_texture(float(threadIdx.x)/float(blockDim.x-1),M);
            float x = tex1D(tex, v);
            //printf("%f\n", x); // for deviceemu testing
            d_out[threadIdx.x] = x;
            printf("(%.2f, %.2f , %d)\n",v-0.5,d_out[threadIdx.x],M);
        }
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 1){
        printf("\n");
    }


    }

int main(){
    int M = 6;
    int nbins = 10;
    int N = M*nbins-1;

    // memory for output

    float *d_out;

    cudaMalloc((void**)&d_out, sizeof(float) * N);


    // make an array half the size of the output

    cudaArray* cuArray;

    cudaMallocArray(&cuArray, &tex.channelDesc, M+1, 1);
    cudaBindTextureToArray (tex, cuArray);

    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = 0;

    // data fill array with increasing values
    float *data = (float*)malloc((M+1)*sizeof(float));

    for (int i = 0; i < (M+1); i++)
        data[i] = float(i);
    ( cudaMemcpyToArray(cuArray, 0, 0, data, sizeof(float)*(M+1), cudaMemcpyHostToDevice) );



    kernel<<<1, nbins+1>>>(M,N, d_out);

    float *h_out = (float*)malloc(sizeof(float)*N);
    ( cudaMemcpy(h_out, d_out, sizeof(float)*N, cudaMemcpyDeviceToHost) );
    /*
    for (int i = 0; i < N; i++)
        printf("%f\n", h_out[i]);
    */

    free(h_out);
    free(data);

    cudaFreeArray(cuArray);
    cudaFree(d_out);
}



