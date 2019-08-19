#include <stdio.h>

__global__ void kernelSampleLayeredTexture(
    cudaTextureObject_t tex,
    int width,
    float* normalized_indices,
    int nlayers){

    // force threads to execute in a specific order
    for (int layer=0; layer < nlayers; layer++){
        for (int i = 0; i < width; i++){
            float u = i+0.5;
            float x = tex1DLayered<float>(tex,u,layer); // last 0 is the 0th layer
            printf("(%.2f, %.2f)\n",u-0.5,x);
        }
    }
}

cudaTextureObject_t simpleLayeredTexture(float * h_data, int width, int num_layers){
    /* h_data is flattened array containing all the information... */
    // allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMemcpy3DParms my_params = {0};
    //my_params.srcPos = make_cudaPos(0,0,0);
    //my_params.dstPos = make_cudaPos(0,0,0);
    my_params.srcPtr = make_cudaPitchedPtr(h_data,width *sizeof(float),width,1);
    my_params.kind = cudaMemcpyHostToDevice;
    my_params.extent = make_cudaExtent(width, 1, num_layers);

    // create the cuda array and copy the data to it
    cudaArray *cu_3darray;
    cudaMalloc3DArray(&cu_3darray, &channelDesc, make_cudaExtent(width, 0,num_layers),cudaArrayLayered);
    my_params.dstArray = cu_3darray;
    cudaMemcpy3D(&my_params);

    // Describe the input array
    cudaResourceDesc            resDesc;
    memset(&resDesc,0,sizeof(cudaResourceDesc));

    resDesc.resType            = cudaResourceTypeArray;
    resDesc.res.array.array    = cu_3darray;

    // Describe the output texture
    cudaTextureDesc             texDesc;
    memset(&texDesc,0,sizeof(cudaTextureDesc));
    texDesc.normalizedCoords = false;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t         tex;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
    return tex;
}

void sampleTexture(
    cudaTextureObject_t tex,
    int width,
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

    cudaTextureObject_t tex = simpleLayeredTexture(data,width, height);
    sampleTexture(tex,width,nsamples,height);

    free(data);
    cudaDestroyTextureObject(tex);
    //cudaFree(cuArray); <--- rip?
}
