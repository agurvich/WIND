extern "C" {
    #include "chimes_vars.h"
    #include "chimes_proto.h"
}

// link to global texture objects defined in wind_chimes.h
#include "wind_chimes.h"

struct wind_chimes_constant_struct wind_chimes_table_constant;
struct wind_chimes_T_dependent_struct wind_chimes_table_T_dependent;
struct wind_chimes_recombination_AB_struct wind_chimes_table_recombination_AB;

void initialize_table_constant(
    struct wind_chimes_constant_struct * p_this_table,
    int * N_reactions,
    int * reactantss_transpose_flat,
    int * productss_transpose_flat,
    int H2_form_heating_reaction_index){

    // bind the simple stuff that's already allocated in the struct
    p_this_table->N_reactions[0] = N_reactions[0];
    p_this_table->N_reactions[1] = N_reactions[1];
    p_this_table->H2_form_heating_reaction_index=H2_form_heating_reaction_index; 

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
}


void initialize_table_T_dependent(
    struct wind_chimes_T_dependent_struct * p_this_table,
    int * N_reactions,
    int * reactantss_transpose_flat,
    int * productss_transpose_flat,
    int H2_collis_dissoc_heating_reaction_index,
    int H2_form_heating_reaction_index){

    // bind the simple stuff that's already allocated in the struct
    p_this_table->N_reactions[0] = N_reactions[0];
    p_this_table->N_reactions[1] = N_reactions[1];
    p_this_table->H2_collis_dissoc_heating_reaction_index=H2_collis_dissoc_heating_reaction_index;
    p_this_table->H2_form_heating_reaction_index=H2_form_heating_reaction_index; 

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
}


void initialize_table_recombination_AB(
    struct wind_chimes_recombination_AB_struct * p_this_table,
    int * N_reactions,
    int * reactantss_transpose_flat,
    int * productss_transpose_flat){

    // bind the simple stuff that's already allocated in the struct
    p_this_table->N_reactions[0] = N_reactions[0];
    p_this_table->N_reactions[1] = N_reactions[1];

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
}

void load_rate_coeffs_into_texture_memory(
    cudaTextureObject_t ** p_p_texture,
    ChimesFloat * texture_edgess_flat,
    int n_layers, // N_reactions_all
    int n_texture_edges // n_texture_edgess
    ){

    // allocate memory on device for these rates constants and 
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
        32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMemcpy3DParms my_params = {0};
    //my_params.srcPos = make_cudaPos(0,0,0);
    //my_params.dstPos = make_cudaPos(0,0,0);
    my_params.srcPtr = make_cudaPitchedPtr(
        texture_edgess_flat,
        n_texture_edges *sizeof(float),// size in bytes
        n_texture_edges, // size in elements
        1); // height dimensionality? (e.g. 1 for 2d?)

    my_params.kind = cudaMemcpyHostToDevice;
    my_params.extent = make_cudaExtent(
        n_texture_edges, // x dim
        1, // y dim
        n_layers); // z dim (layers)

    // create the cuda array and copy the data to it
    cudaArray *cu_3darray;
    cudaMalloc3DArray(
        &cu_3darray,
        &channelDesc,
        make_cudaExtent(
            n_texture_edges, // x dim
            0, // y dim -- according to stack overflow this needs to be 0  even though above it is 1
            n_layers), // z dim
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
        *(p_p_texture),// was passed the pointer to the texture by reference
        &resDesc,
        &texDesc,
        NULL);
}

void tranpose_flatten_chemical_equations(
    int ** chem_indices,
    int ** p_chem_indices_transpose_flat,
    int N_reactions,
    int N_chems){

    for (int i_chem=0; i_chem<N_chems; i_chem++){
        for (int i_rxn=0; i_rxn<N_reactions; i_rxn++){
            (*p_chem_indices_transpose_flat)[i_chem*N_reactions+i_rxn] = chem_indices[i_rxn][i_chem];
        }
    }
}

void create_wind_chimes_structs(){
    ChimesFloat * ratess_flat;
    int * reactantss_transpose_flat;
    int * productss_transpose_flat;
    int N_reactions_all;
/* ------- chimes_table_constant ------- */
    N_reactions_all = chimes_table_constant.N_reactions[1];

    // allocate the pointers for this table
    ratess_flat = (ChimesFloat *) malloc(sizeof(ChimesFloat)*N_reactions_all);
    reactantss_transpose_flat = (int *) malloc(sizeof(int)*3*N_reactions_all);
    productss_transpose_flat = (int *) malloc(sizeof(int)*3*N_reactions_all);

    tranpose_flatten_chemical_equations(
        chimes_table_constant.reactants,
        &reactantss_transpose_flat, 
        N_reactions_all,
        3); // 3 reactants per reaction

    tranpose_flatten_chemical_equations(
        chimes_table_constant.products,
        &productss_transpose_flat, 
        N_reactions_all,
        3); // 3 products per reaction

    // read the values from the corresponding chimes_table
    initialize_table_constant(
        &wind_chimes_table_constant,
        chimes_table_constant.N_reactions,
        reactantss_transpose_flat,
        productss_transpose_flat,
        chimes_table_constant.H2_form_heating_reaction_index);

    // allocate the memory for the constant rates on the device
    //  which are just an array, no interpolation required
    cudaMalloc(
        &(wind_chimes_table_constant.rates),
        sizeof(ChimesFloat)*N_reactions_all);
    cudaMemcpy(
        wind_chimes_table_constant.rates,
        chimes_table_constant.rates,
        sizeof(ChimesFloat)*N_reactions_all,
        cudaMemcpyHostToDevice);
    // and free up the ratess, productss, and reactantss buffers
    free(reactantss_transpose_flat);
    free(productss_transpose_flat);
    free(ratess_flat);

/* ------- chimes_table_T_dependent ------- */
    N_reactions_all = chimes_table_T_dependent.N_reactions[1];

    // (re-)allocate the pointers for this table
    ratess_flat = (ChimesFloat *) malloc(
        sizeof(ChimesFloat)*
        chimes_table_bins.N_Temperatures*
        N_reactions_all);
    reactantss_transpose_flat = (int *) malloc(sizeof(int)*3*N_reactions_all);
    productss_transpose_flat = (int *) malloc(sizeof(int)*3*N_reactions_all);

    tranpose_flatten_chemical_equations(
        chimes_table_T_dependent.reactants,
        &reactantss_transpose_flat, 
        N_reactions_all,
        3); // 3 reactants per reaction

    tranpose_flatten_chemical_equations(
        chimes_table_T_dependent.products,
        &productss_transpose_flat, 
        N_reactions_all,
        3); // 3 products per reaction

    // copy the values from the table over...
    initialize_table_T_dependent(
        &wind_chimes_table_T_dependent,
        chimes_table_T_dependent.N_reactions,
        reactantss_transpose_flat,
        productss_transpose_flat,
        chimes_table_T_dependent.H2_collis_dissoc_heating_reaction_index,
        chimes_table_T_dependent.H2_form_heating_reaction_index);

    // TODO need to make sure rates is in the right format TODO
    // read the rate coeffs from the corresponding chimes_table
    //  and put them into texture memory
    load_rate_coeffs_into_texture_memory(
        &wind_chimes_table_T_dependent.rates,
        ratess_flat,
        N_reactions_all,
        chimes_table_bins.N_Temperatures);

    // free up the ratess, productss, and reactantss buffers
    free(reactantss_transpose_flat);
    free(productss_transpose_flat);
    free(ratess_flat);
/* ------- chimes_table_recombination_AB ------- */

    // (re-)allocate the pointers for this table
    ratess_flat = (ChimesFloat *) malloc(
        sizeof(ChimesFloat)*
        chimes_table_bins.N_Temperatures*
        N_reactions_all*
        2); // 2x the rates, one for A, one for B

    reactantss_transpose_flat = (int *) malloc(sizeof(int)*3*N_reactions_all);
    productss_transpose_flat = (int *) malloc(sizeof(int)*3*N_reactions_all);

    tranpose_flatten_chemical_equations(
        chimes_table_recombination_AB.reactants,
        &reactantss_transpose_flat, 
        N_reactions_all,
        3); // 3 reactants per reaction

    // only 1 product per reaction -> this is already in the format we need!
    productss_transpose_flat = chimes_table_recombination_AB.products;

    // copy the values from the table over...
    initialize_table_recombination_AB(
        &wind_chimes_table_recombination_AB,
        chimes_table_recombination_AB.N_reactions,
        reactantss_transpose_flat,
        productss_transpose_flat);

    // TODO need to make sure rates is in the right format TODO
    // read the rate coeffs from the corresponding chimes_table
    //  and put them into texture memory
    load_rate_coeffs_into_texture_memory(
        &wind_chimes_table_recombination_AB.rates,
        ratess_flat,
        2*N_reactions_all, // 2x N_reactions_all layers, first half for A second for B
        chimes_table_bins.N_Temperatures);

    // for looping through mallocs on the device...
    //https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dynamic-global-memory-allocation-and-operations
}

// to unmangle the name, since I can
extern "C" {
    void init_wind_chimes(struct globalVariables * myGlobalVars){
        // call the existing C routine...
        init_chimes(myGlobalVars);

        create_wind_chimes_structs();
    }
}

void init_chimes_wind_hardcoded(struct globalVariables myGlobalVars){
    // use hardcoded rates arrays: 
    create_wind_chimes_structs();
}
