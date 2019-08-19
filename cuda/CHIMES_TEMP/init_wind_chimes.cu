#include "chimes_vars.h"
#include "chimes_proto.h"

// link to global texture objects defined in wind_chimes.h
#include "wind_chimes.h"


void load_rate_coeffs_into_texture_memory(){
    // do some magic that will add the necessary info to the global textures
    
/* ------- chimes_table_constant ------- */
    // allocate the memory for the constant rates on the device
    cudaMalloc(
        &wind_chimes_table_constant,
        sizeof(ChimesFloat)*chimes_table_constant.N_reactions[1]);

    // copy it over
    cudaMemcpy(
        wind_chimes_table_constant,
        chimes_table_constant.rates,
        sizeof(ChimesFloat)*chimes_table_constant.N_reactions[1],
        cudaMemcpyHostToDevice)

    // TODO need to copy over the reaction info? the number of reactions?
    //  need to figure out how I will represent this on device anyway sigh.
    //  might need to just make a dang struct and live with it TODO
        
/* ------- chimes_table_T_dependent ------- */
    // copy the rates over into an array...
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
        32,0,0,0,
        cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    // cudaMallocArray inherently allocates 2D, 
    //  last argument is 
    cudaMallocArray(
        &cuArray,
        &channelDesc,
        Narr,
        1);

    // TODO need to loop over reactions and make sure
    //  rates are stored how I think they are in memory
    //  before copying them to Array TODO

    // cudaMemcpyToArray is deprecated for some reason...
    //  so we're supposed to be using Memcpy2DToArray
    //  https://devtalk.nvidia.com/default/topic/1048376/cuda-programming-and-performance/cudamemcpytoarray-is-deprecated/
    cudaMemcpyToArray(
        cuArray, // destination of data
        0,0, // woffset and hoffset?
        arr, // source of data
        sizeof(float)*Narr, // bytes of data
        cudaMemcpyHostToDevice);

    // TODO define descriptors for the texture rates TODO 


    // create the texture object
    cudaCreateTextureObject(
        &wind_chimes_table_T_dependent,
        &resDesc,
        &texDesc,
        NULL);
}

void init_chimes_wind(struct globalVariables myGlobalVars){

    // call the existing C routine...
    init_chimes_wind(myGlobalVars);

    load_rate_coeffs_into_texture_memory();
    

    
}

void init_chimes_wind_hardcoded(struct globalVariables myGlobalVars){
    // use hardcoded rates arrays: 

    load_rate_coeffs_into_texture_memory();
}
