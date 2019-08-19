#include <stdio.h>

__global__ void kernelSampleLayeredTexture(
    cudaTextureObject_t tex,
    int width,
    int height,
    float* normalized_indices,
    int nlayers){

    // force threads to execute in a specific order
    for (int i = 0; i < width; i++){
        for(int j=0; j < height; j++){
            float u = i+0.5;
            float v = j+0.5;
            float x = tex2D<float>(tex,u,v);
            printf("(%.2f, %.2f, %.2f)\n",u-0.5,v-0.5,x);
        }
    }
}

/*
    myparms.dstArray = cu_3darray;
    myparms.extent = make_cudaExtent(
        width,
        height,
        num_layers);
    myparms.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&myparms));

*/

cudaTextureObject_t simpleLayeredTexture(float * h_data, int width, int height, int num_layers){
    /* h_data is flattened array containing all the information... */
    // allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cu_3darray;
    cudaMalloc3DArray(&cu_3darray, &channelDesc, make_cudaExtent(width, height,0)); //num_layers, cudaArrayLayered));
    cudaMemcpy3DParms myparms = {0};
    myparms.srcPos = make_cudaPos(0,0,0);
    myparms.dstPos = make_cudaPos(0,0,0);
    myparms.srcPtr = make_cudaPitchedPtr(h_data, width * sizeof(float), width, height);
    myparms.dstArray = cu_3darray;
    myparms.extent = make_cudaExtent(width, height, 0);//num_layers);
    myparms.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&myparms);

    cudaTextureObject_t         tex;
    cudaResourceDesc            texRes;
    memset(&texRes,0,sizeof(cudaResourceDesc));

    texRes.resType            = cudaResourceTypeArray;
    texRes.res.array.array    = cu_3darray;

    cudaTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL);
    return tex;
}

/*
cudaTextureObject_t make2DLayeredTextureFromPointer(
    float * arr,
    int Narr){
    
    int height=1;
    int num_layers=1;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
        32,0,0,0,
        cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    // cudaMallocArray inherently allocates 2D, 
    //  last argument is 
    cudaMalloc3DArray(
        &cuArray,
        &channelDesc,
        make_cudaExtent(Narr, height, num_layers));
        //cudaArrayLayered);

    cudaMemcpy3DParms myparms = {0};
    myparms.srcPos = make_cudaPos(0,0,0);
    myparms.dstPos = make_cudaPos(0,0,0);
    myparms.srcPtr = make_cudaPitchedPtr(
        arr,
        Narr*sizeof(float),
        Narr,
        height);
    myparms.dstArray = cuArray;
    myparms.extent = make_cudaExtent(
        Narr,
        height,
        num_layers);
    myparms.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&myparms);

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
*/

void sampleTexture(
    cudaTextureObject_t tex,
    int width,
    int height,
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
    kernelSampleLayeredTexture<<<1, 1>>>(
        tex,
        width,
        height,
        d_normalized_indices,
        nlayers);

    cudaDeviceSynchronize();
    free(normalized_indices);
    cudaFree(d_normalized_indices);
}

int main(){

    int nsamples = 10; // sample the texture in 1/nsamples increments
    int width = 5;
    int height = 5;
    int ntexture_edges = width*height; // how many "anchors" are in the texture

    float xc = 2;
    float yc = 2;
    // fill array with texture values
    float *data = (float*)malloc(ntexture_edges*sizeof(float));
    for (int i =0; i < width; i++){
        for (int j = 0; j < height; j++){
            data[i+j*width] = (i-xc)*(i-xc) + (j-yc)*(j-yc);//2*float(i);
            printf("%.1f\t",data[i+j*width]);
        }
        printf("\n");
    }

    //cudaTextureObject_t tex = make2DLayeredTextureFromPointer(
        //data,ntexture_edges);

    cudaTextureObject_t tex = simpleLayeredTexture(data,ntexture_edges/2, 2, 1);
    sampleTexture(tex,width,height,nsamples,1);

    free(data);
    cudaDestroyTextureObject(tex);
    //cudaFree(cuArray); <--- rip?
}
