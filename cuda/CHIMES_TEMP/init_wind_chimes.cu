#include "chimes_vars.h"
#include "chimes_proto.h"

// link to global texture objects defined in wind_chimes.h
#include "wind_chimes.h"

void initialize_table(
    struct wind_chimes_T_dependent_struct * p_this_table,
    int * N_reactions,
    int * reactantss_transpose_flat,
    int * productss_transpose_flat,
    int H2_collis_dissoc_heating_reaction_index,
    int H2_form_heating_reaction_index,
    void * rates){

    // bind the simple stuff that's already allocated in the struct
    p_this_table->N_reactions[0] = N_reactions[0];
    p_this_table->N_reactions[1] = N_reactions[1];
    p_this_table->H2_collis_dissoc_heating_reaction_index=H2_collis_dissoc_heating_reaction_index;
    p_this_table->H2_form_heating_reaction_index=H2_form_heating_reaction_index; 

    printf(" n reactions is %d %d \n",p_this_table->N_reactions[0],p_this_table->N_reactions[1]);

    // allocate and copy the reactants over, then bind to the host this_table structure
    int * d_reactantss_transpose_flat;
    cudaMalloc(&d_reactantss_transpose_flat,
        sizeof(int)*3*p_this_table->N_reactions[1]);
    cudaMemcpy(d_reactantss_transpose_flat,
        reactantss_transpose_flat,
        sizeof(int)*3*p_this_table->N_reactions[1],
        cudaMemcpyHostToDevice);
    p_this_table->reactantss_transpose_flat = d_reactantss_transpose_flat; // needs to be a device array

    // allocate and copy the products over, then bind to the host this_table structure
    int * d_productss_transpose_flat;
    cudaMalloc(
        &d_productss_transpose_flat,
        sizeof(int)*3*p_this_table->N_reactions[1]);
    cudaMemcpy(
        d_productss_transpose_flat,
        productss_transpose_flat,
        sizeof(int)*3*p_this_table->N_reactions[1],
        cudaMemcpyHostToDevice);
    p_this_table->productss_transpose_flat = d_productss_transpose_flat; // needs to be a device array
    // allocate and copy the rates over, then bind to the host table structure

    float * d_rates;
    cudaMalloc(
        &d_rates,
        sizeof(float)*p_this_table->N_reactions[1]);
    cudaMemcpy(
        d_rates,
        rates,
        sizeof(float)*p_this_table->N_reactions[1],
        cudaMemcpyHostToDevice);
    p_this_table->rates = d_rates; // needs to be a device array
}

void load_rate_coeffs_into_texture_memory(void * ){

}

void create_wind_chimes_structs(){
    // do some magic that will add the necessary info to the global textures
/* ------- chimes_table_constant ------- */
    // read the values from the corresponding chimes_table
    int N_reactions_all = chimes_table_constant.N_reactions[1];
    float * ratess_flat = chimes_table_constant.rates; // 1xN_reactions_all, not log

    // allocate the memory for the constant rates on the device
    cudaMalloc(
        &(wind_chimes_table_constant.rates),
        sizeof(ChimesFloat)*chimes_table_constant.N_reactions[1]);

    // copy it over
    cudaMemcpy(
        wind_chimes_table_constant.rates,
        chimes_table_constant.rates,
        sizeof(ChimesFloat)*chimes_table_constant.N_reactions[1],
        cudaMemcpyHostToDevice)

    // TODO need to copy over the reaction info? the number of reactions?
    //  need to figure out how I will represent this on device anyway sigh.
    //  might need to just make a dang struct and live with it TODO

    initialize_table(
        &wind_chimes_table_constant,
        chimes_table_constant.N_reactions,
        int * reactantss_transpose_flat,
        int * productss_transpose_flat,
        chimes_table_constant.H2_collis_dissoc_heating_reaction_index,
        chimes_table_constant.H2_form_heating_reaction_index,
        (void *) chimes_table_constant.rates);

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

    // create the texture object at the global memory structure
    cudaCreateTextureObject(
        &(wind_chimes_table_T_dependent.rates),
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

    create_wind_chimes_structs();
}

void init_chimes_wind_hardcoded(struct globalVariables myGlobalVars){
    // use hardcoded rates arrays: 
    create_wind_chimes_structs();
}
