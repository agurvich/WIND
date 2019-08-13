#include <stdio.h>

//texture<float, 1, cudaReadModeElementType> tex;

__global__ void kernel(
    cudaTextureObject_t tex,
    int texture_size,
    float* normalized_indices){
    for (int i = 0; i < blockDim.x; i++){
        if (threadIdx.x == i){
            float v=0.5+(normalized_indices[threadIdx.x]*
                (texture_size-1));
            float x = tex1D<float>(tex, v);
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

cudaTextureObject_t make1DTextureFromPointer(
    float * arr,
    int Narr){
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
        32,0,0,0,
        cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, Narr, 1);

    cudaMemcpyToArray(cuArray,0,0,arr,sizeof(float)*Narr,
        cudaMemcpyHostToDevice);

    //create texture object
    struct cudaResourceDesc resDesc;
    memset(&resDesc,0,sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] =cudaAddressModeClamp;
    texDesc.addressMode[1] =cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

      // create texture object: we only have to do this once!
    cudaTextureObject_t tex=0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
    return tex;
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
    
    cudaTextureObject_t tex = make1DTextureFromPointer(
        data,ntexture_edges);

    kernel<<<1, nsamples+1>>>(
        tex,
        ntexture_edges,
        d_normalized_indices);

    cudaDeviceSynchronize();

    free(data);
    free(normalized_indices);

    cudaDestroyTextureObject(tex);
    //cudaFreeArray(cuArray);
    cudaFree(d_normalized_indices);
}



