#include <stdio.h>

__global__ void kernelSampleLayeredTexture(
    cudaTextureObject_t tex,
    int texture_size,
    float* normalized_indices,
    int nlayers){

    for (int layer=0; layer<nlayers; layer++){
        // force threads to execute in a specific order
        for (int i = 0; i < blockDim.x; i++){
            if (threadIdx.x == i){
                float v=0.5+(normalized_indices[threadIdx.x]*
                    (texture_size-1));
                //float x = tex1D<float>(tex, v);
                float x = v;
                printf("(%.2f, %.2f, %.2f)\n",
                    normalized_indices[threadIdx.x],
                    v-0.5,x);
            }
            __syncthreads();
        }
    }
}

/*
    cudaArray *cu_3darray;
    cudaMalloc3DArray(
        &cu_3darray,
        &channelDesc,
        make_cudaExtent(width, height, num_layers),
        cudaArrayLayered);

    cudaMemcpy3DParms myparms = {0};
    myparms.srcPos = make_cudaPos(0,0,0);
    myparms.dstPos = make_cudaPos(0,0,0);
    myparms.srcPtr = make_cudaPitchedPtr(
        h_data,
        width*sizeof(float),
        width,
        height);

    myparms.dstArray = cu_3darray;
    myparms.extent = make_cudaExtent(
        width,
        height,
        num_layers);
    myparms.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&myparms));

    cudaTextureObject_t         tex;
    cudaResourceDesc            texRes;
    memset(&texRes,0,sizeof(cudaResourceDesc));

    texRes.resType            = cudaResourceTypeArray;
    texRes.res.array.array    = cu_3darray;

    cudaTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode       = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));

*/




cudaTextureObject_t make1DTextureFromPointer(
    float * arr,
    int Narr){
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
        32,0,0,0,
        cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    // cudaMallocArray inherently allocates 2D, 
    //  last argument is 
    cudaMalloc3DArray(
        &cuArray,
        &channelDesc,
        make_cudaExtent(Narr, 0, 0));

    // cudaMemcpyToArray is deprecated for some reason...
    //  so we're supposed to be using Memcpy2DToArray
    //  https://devtalk.nvidia.com/default/topic/1048376/cuda-programming-and-performance/cudamemcpytoarray-is-deprecated/
    cudaMemcpyToArray(
        cuArray, // destination of data
        0,0, // woffset and hoffset?
        arr, // source of data
        sizeof(float)*Narr, // bytes of data
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

void sampleTexture(
    cudaTextureObject_t tex,
    int ntexture_edges,
    int nsamples,
    int nlayers){

    // create the normalized_indices
    //  +1 to get the final 1.0 at the end
    float *normalized_indices = (float*)malloc(
        (nsamples+1)*sizeof(float));

    for (int i=0; i<(nsamples+1); i++){
        normalized_indices[i] = float(i)/float(nsamples);
    }

    // create the device normalized_indices pointer and fill it
    float * d_normalized_indices;
    cudaMalloc((void**)&d_normalized_indices,
        sizeof(float)*(nsamples+1));
    cudaMemcpy(d_normalized_indices,
        normalized_indices,
        sizeof(float)*(nsamples+1),
        cudaMemcpyHostToDevice);
    kernelSampleLayeredTexture<<<1, nsamples+1>>>(
        tex,
        ntexture_edges,
        d_normalized_indices,
        nlayers);

    cudaDeviceSynchronize();
    free(normalized_indices);
    cudaFree(d_normalized_indices);
}

int main(){
    int ntexture_edges = 6; // how many "anchors" are in the texture
    int nsamples = 10; // sample the texture in 1/nsamples increments

    // fill array with texture values
    float *data = (float*)malloc(ntexture_edges*sizeof(float));
    for (int i = 0; i < ntexture_edges; i++){
        data[i] = 2*float(i);
        printf("%d\t",i);
    }
    printf("\n");
    
    cudaTextureObject_t tex = make1DTextureFromPointer(
        data,ntexture_edges);

    sampleTexture(tex,ntexture_edges,nsamples,1);

    free(data);
    cudaDestroyTextureObject(tex);
    //cudaFree(cuArray); <--- rip?
}
