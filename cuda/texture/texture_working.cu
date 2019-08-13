#include <stdio.h>

texture<float, 1, cudaReadModeElementType> tex;

__global__ void kernel(
    int texture_size,
    float* normalized_indices){
    for (int i = 0; i < blockDim.x; i++){
        if (threadIdx.x == i){
            float v=0.5+normalized_indices[threadIdx.x]*(
                texture_size-1);
            float x = tex1D(tex, v);
            printf("(%.2f, %.2f, %.2f)\n",
                normalized_indices[threadIdx.x],
                v-0.5,x);
        }
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 1){
        printf("\n");
    }
}

int main(){
    int ntexture_edges = 6; // how many "anchors" are in the texture
    int nsamples = 10; // sample the texture in 1/nsamples increments

    // create the normalized_indices
    //  +1 to get the final 1.0 at the end
    float *normalized_indices = (float*)malloc(
        (nsamples+1)*sizeof(float));

    for (int i=0; i<(nsamples+1); i++){
        normalized_indices[i] = float(i)/float(nsamples);
    }
    normalized_indices[3]=.75; // overwrite to check manually

    // create the device normalized_indices pointer and fill it
    float * d_normalized_indices;
    cudaMalloc((void**)&d_normalized_indices,
        sizeof(float)*(nsamples+1));
    cudaMemcpy(d_normalized_indices,
        normalized_indices,
        sizeof(float)*(nsamples+1),
        cudaMemcpyHostToDevice);

    // fill array with texture values
    float *data = (float*)malloc(ntexture_edges*sizeof(float));
    for (int i = 0; i < ntexture_edges; i++){
        data[i] = 2*float(i);
        printf("%d\t",i);
    }
    printf("\n");

    // make an array to store the texture values
    cudaArray* cuArray;

    cudaMallocArray(&cuArray, &tex.channelDesc, ntexture_edges, 1);
    cudaBindTextureToArray (tex, cuArray);
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = 0;


    cudaMemcpyToArray(cuArray,0,0,data,sizeof(float)*ntexture_edges,
        cudaMemcpyHostToDevice);

    kernel<<<1, nsamples+1>>>(
        ntexture_edges,
        d_normalized_indices);

    cudaDeviceSynchronize();

    free(data);
    free(normalized_indices);

    cudaFreeArray(cuArray);
    cudaFree(d_normalized_indices);
}



