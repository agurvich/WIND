#include "chimes_vars.h"
#include "chimes_proto.h"

// link to global texture objects defined in wind_chimes.h
#include "wind_chimes.h"


void load_rate_coeffs_into_texture_memory(){
/* ------- chimes_table_constant ------- */
    // read the values from the corresponding chimes_table
    int N_reactions_all = chimes_table_constant.N_reactions[1];
    float * ratess_flat = chimes_table_constant.rates; // 1xN_reactions_all, not log

    // allocate the memory for the constant rates on the device
    cudaMalloc(
        &wind_chimes_table_constant,
        sizeof(ChimesFloat)*N_reactions_all);

    // copy it over
    cudaMemcpy(
        wind_chimes_table_constant,
        ratess_flat,
        sizeof(ChimesFloat)*N_reactions_all,
        cudaMemcpyHostToDevice)

    // TODO need to copy over the reaction info
    N_reactions[0] and N_reactions[1] 
    reactantss // need to take the transpose
    productss // need to take the transpose

    H2_form_heating_reaction_index
        
/* ------- chimes_table_T_dependent ------- */
    // read the values from the corresponding chimes_table
    N_reactions_all = chimes_table_T_dependent.N_reactions[1];
    // put the flat pointer at the head of the 2d array
    ratess_flat = chimes_table_T_dependent.rates[0];

    // TODO need to loop over reactions and make sure
    //  rates are stored how I think they are in memory
    //  before copying them to Array TODO


    // allocate memory on device for these rates constants and 
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMemcpy3DParms my_params = {0};
    //my_params.srcPos = make_cudaPos(0,0,0);
    //my_params.dstPos = make_cudaPos(0,0,0);
    my_params.srcPtr = make_cudaPitchedPtr(ratess_flat,N_Temperature *sizeof(float),N_Temperature,1);
    my_params.kind = cudaMemcpyHostToDevice;
    my_params.extent = make_cudaExtent(N_Temperature, 1, N_reactions_all);

    // create the cuda array and copy the data to it
    cudaArray *cu_3darray;
    cudaMalloc3DArray(
        &cu_3darray,
        &channelDesc,
        make_cudaExtent(N_Temperature, 0,N_reactions_all),
        cudaArrayLayered);
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
    texDesc.normalizedCoords = true;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(
        &wind_chimes_table_T_dependent,
        &resDesc,
        &texDesc,
        NULL);

    // TODO need to copy over the reaction info
    N_reactions[0] and N_reactions[1]
    reactantss
    productss

    H2_collis_dissoci_heating_reaction_index
    H2_form_heating_reaction_index


    // for looping through mallocs on the device...
    //https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dynamic-global-memory-allocation-and-operations
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
