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


    // create the texture object at the global memory structure
    cudaCreateTextureObject(
        &(wind_chimes_table_T_dependent.rates),
        &resDesc,
        &texDesc,
        NULL);
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
