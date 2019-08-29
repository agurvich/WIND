#include <math.h>

#include "config.h"
#include "ode.h"
#include "wind_chimes.h"


__global__ void read_texture(void * input){

    // cast to the correct format
    struct RHS_input_struct * p_RHS_input = (struct RHS_input_struct *) input;

    // unpack the RHS_input struct
    struct wind_chimes_constant_struct * p_wind_chimes_table_constant = p_RHS_input->table_constant;
    struct wind_chimes_T_dependent_struct * p_wind_chimes_table_T_dependent = p_RHS_input->table_T_dependent;
    struct wind_chimes_recombination_AB_struct * p_wind_chimes_table_recombination_AB = p_RHS_input->table_recombination_AB;
    struct wind_chimes_table_bins_struct * p_wind_chimes_table_bins = p_RHS_input->table_bins;

    // hard code all reactions, TODO Tmol switch
    int N_reactions = p_wind_chimes_table_T_dependent->N_reactions[1];
    int N_Temperatures = p_wind_chimes_table_bins->N_Temperatures;

    cudaTextureObject_t tex = p_wind_chimes_table_recombination_AB->rates;

    N_reactions=4; // TODO DEBUG 

    // loop through each of the reactions
    for (int rxn_i=0; rxn_i < N_reactions; rxn_i++){
        // print out each temperature
        for (int i = 0; i < N_Temperatures; i++){
            float u = i+0.5;
            float x = tex1DLayered<float>(tex,u,rxn_i); // last 0 is the 0th layer
            printf("%.4f,",x);
        }
        printf("]\n");
    }
}

__device__ void propagate_rate_coeff(
    ChimesFloat this_rate_coeff,
    ChimesFloat nH,
    int tid,
    int N_reactions,
    int * reactantss,
    int N_reactants,
    int * productss,
    int N_products,
    WindFloat * shared_equations,
    WindFloat * shared_dydts,
    WindFloat * Jacobians){

   // Devices of compute capability 3.x have configurable bank size, which can be set using cudaDeviceSetSharedMemConfig() to either four bytes (cudaSharedMemBankSizeFourByte, the default) or eight bytes (cudaSharedMemBankSizeEightByte). Setting the bank size to eight bytes can help avoid shared memory bank conflicts when accessing double precision data. 

    WindFloat this_partial;
    WindFloat this_abundance;
    int this_react;
    // calculate the rate itself by multiplying by reactant abundances and 
    //  factors of nH
    for (int reactant_i=0; reactant_i<N_reactants; reactant_i++){
        this_react = reactantss[tid + reactant_i*N_reactions];
        if (this_react>0) this_rate_coeff*=nH*shared_equations[this_react];
    }
    this_rate_coeff/=nH; // want Nreacts-1 many factors of nH

    int this_prod;
    // update creation rates in shared_dydts
    for (int prod_i=0; prod_i<N_products; prod_i++){
        this_prod = productss[tid + prod_i*N_reactions];
        if (this_prod >0) {
            atomicAdd(&shared_dydts[this_prod],this_rate_coeff);// TODO yikes no atomic add for doubles??
        }
    }

    // update destruction rates in shared_dydts (and Jacobian if necessary)
    int jindex;
    // TODO could we stage the column in shared or thread-private memory?
    //  doesn't look like i have a spare shared array (using both 
    //  shared_equations for looking up abundances and
    //  shared_dydts for accumulation)

    //  but... since we're just using shared_equations for look-up... what if 
    //  each thread pulls a copy from shared_equations at the beginning and then thread 0 
    //  refills it after we're done?


    int temp_react,temp_prod;
    for (int reactant_i=0; reactant_i<N_reactants; reactant_i++){
        this_react = reactantss[tid + reactant_i*N_reactions];
        if (this_react >0){
            // subtract it from the total rate
            atomicAdd(&shared_dydts[this_react],-this_rate_coeff); // TODO yikes no atomic add for doubles??
#ifdef SIE
            // take the partial derivative w.r.t to this reactant abundance
            this_abundance = shared_equations[this_react];
            if (this_abundance>0) this_partial = this_rate_coeff/this_abundance;
            else this_partial=0;

            // update rows of reactant Jacobian with partial destruction rate
            for (int temp_react_i=0; temp_react_i<N_reactants; temp_react_i++){
                temp_react = reactantss[tid + temp_react_i*N_reactions];
                jindex = this_react*blockDim.x + temp_react; // J is in column-major-order
                if (temp_react >0) atomicAdd(&Jacobians[jindex],-this_partial); // TODO yikes no atomic add for doubles??
            } // for temp_react in reacts

            // update rows of product jacobian with partial creation rate
            for (int temp_prod_i=0; temp_prod_i<N_products; temp_prod_i++){
                temp_prod = productss[tid + temp_prod_i*N_reactions];
                jindex = this_react*blockDim.x + temp_prod; // J is in column-major-order
                if (temp_prod >0) atomicAdd(&Jacobians[jindex],this_partial); // TODO yikes no atomic add for doubles??
            } // for temp_prod in prods 
#endif
        }// if this_react > 0 
    }// for react in reacts 
}

__device__ float determine_interpolation_range(
    ChimesFloat value,
    ChimesFloat * edges,
    int nedges){

    ChimesFloat low,high;
    
    // figure out which bin this particle lives in
    //  TODO can this be done in parallel...? or anything better
    //  than every thread looping through this...
    int this_index=0;
    while (this_index < (nedges-1)){
        high = edges[this_index+1]; 
        if (high>value) break;
        this_index++;
    }

    low = edges[this_index];

    // calculate the effective index, offset by 0.5 because texels are 
    //  located at bin centers...
    return (float) 0.5 + this_index + (value-low)/(high-low);
}

__device__ void loop_over_reactions_constant(
    int N_reactions,
    int * reactantss_transpose_flat,
    int N_reactants,
    int * productss_transpose_flat,
    int N_products,
    ChimesFloat * rate_coeffs,
    ChimesFloat nH,
    WindFloat * shared_equations,
    WindFloat * shared_dydts,
    WindFloat * Jacobians){

    int tid;
    ChimesFloat this_rate_coeff;
    for (int rxn_i=0; rxn_i < (N_reactions/blockDim.x+1); rxn_i++){
        // do blockDim.x many reactions simultaneously
        tid = rxn_i*blockDim.x + threadIdx.x;

        // do we have more reactions to solve?
        if (tid < N_reactions){
            // read rate directly from array
            this_rate_coeff = rate_coeffs[tid]*3.15e7; //1/yr

            // put this rate coefficient's contribution in the relevant
            //  rate arrays and jacobian entries
            propagate_rate_coeff(
                this_rate_coeff,
                nH,
                tid,
                N_reactions,
                reactantss_transpose_flat,
                N_reactants,
                productss_transpose_flat,
                N_products,
                shared_equations,
                shared_dydts,
                Jacobians);
        } // if tid < N_reactions
    } // for rxn_i in N_reactions/blockDim.x
}

__device__ void loop_over_reactions_T_dependent(
    int N_reactions,
    int * reactantss_transpose_flat,
    int N_reactants,
    int * productss_transpose_flat,
    int N_products,
    cudaTextureObject_t rate_coeff_tex,
    ChimesFloat nH,
    WindFloat * shared_equations,
    WindFloat * shared_dydts,
    WindFloat * Jacobians,
    float T_tex_coord,
    int layer_offset){

    int tid;
    ChimesFloat this_rate_coeff;
    for (int rxn_i=0; rxn_i < (N_reactions/blockDim.x+1); rxn_i++){
        // do blockDim.x many reactions simultaneously
        tid = rxn_i*blockDim.x + threadIdx.x;

        // do we have more reactions to solve?
        if (tid < N_reactions){
            // read off the rate coefficient from the texture
            this_rate_coeff = pow(10.0,
                (ChimesFloat) tex1DLayered<float>(
                    rate_coeff_tex,T_tex_coord,layer_offset+tid))*3.15e7; // 1/yr

            // put this rate coefficient's contribution in the relevant
            //  rate arrays and jacobian entries
            propagate_rate_coeff(
                this_rate_coeff,
                nH,
                tid,
                N_reactions,
                reactantss_transpose_flat,
                N_reactants,
                productss_transpose_flat,
                N_products,
                shared_equations,
                shared_dydts,
                Jacobians);
        } // if tid < N_reactions
    } // for rxn_i in N_reactions/blockDim.x
}

__device__ WindFloat evaluate_RHS_function(
    float tnow,
    void * RHS_input,
    WindFloat * constants,
    WindFloat * shared_equations,
    WindFloat * shared_dydts,
    WindFloat * Jacobians,
    int Nequations_per_system){

    // zero out the derivative vector and jacobian matrix
    shared_dydts[threadIdx.x]=0;

#ifdef SIE
    for (int i=0; i<blockDim.x; i++){
        Jacobians[i*blockDim.x + threadIdx.x]=0;
    }
#endif

    // cast input struct to the correct format
    struct RHS_input_struct * p_RHS_input = (struct RHS_input_struct *) RHS_input;

    // unpack the RHS_input struct
    struct wind_chimes_constant_struct * p_wind_chimes_table_constant = p_RHS_input->table_constant;
    struct wind_chimes_T_dependent_struct * p_wind_chimes_table_T_dependent = p_RHS_input->table_T_dependent;
    struct wind_chimes_recombination_AB_struct * p_wind_chimes_table_recombination_AB = p_RHS_input->table_recombination_AB;
    struct wind_chimes_table_bins_struct * p_wind_chimes_table_bins = p_RHS_input->table_bins;

    // constantss_flat = [T0,nH0,T1,nH1,T2,nH2...] and constants = &constantss_flat[NUM_CONST*blockIdx.x]
    WindFloat logTemperature = constants[0];
    WindFloat nH = constants[1]; 

    WindFloat logTMOL = 2.01;
    
/* ------- chimes_table_constant ------- */
    loop_over_reactions_constant(
        p_wind_chimes_table_constant->N_reactions[logTemperature<logTMOL],
        p_wind_chimes_table_constant->reactantss_transpose_flat,2,
        p_wind_chimes_table_constant->productss_transpose_flat,3,
        p_wind_chimes_table_constant->rates,
        nH,
        shared_equations,
        shared_dydts,
        Jacobians);

/* ------- chimes_table_T_dependent ------- */
    float T_tex_coord = determine_interpolation_range(
        logTemperature,
        p_wind_chimes_table_bins->Temperatures,
        p_wind_chimes_table_bins->N_Temperatures);
    
    loop_over_reactions_T_dependent(
        p_wind_chimes_table_T_dependent->N_reactions[logTemperature<logTMOL],
        p_wind_chimes_table_T_dependent->reactantss_transpose_flat,3,
        p_wind_chimes_table_T_dependent->productss_transpose_flat,3,
        p_wind_chimes_table_T_dependent->rates,
        nH,
        shared_equations,
        shared_dydts,
        Jacobians,
        T_tex_coord,
        0);//int layer_offset

/* ------- chimes_table_recombination_AB ------- */
    loop_over_reactions_T_dependent(
        p_wind_chimes_table_recombination_AB->N_reactions[logTemperature<logTMOL],
        p_wind_chimes_table_recombination_AB->reactantss_transpose_flat,2,
        p_wind_chimes_table_recombination_AB->productss_transpose_flat,1,
        p_wind_chimes_table_recombination_AB->rates,
        nH,
        shared_equations,
        shared_dydts,
        Jacobians,
        T_tex_coord,
        0); //int layer_offset -- case B recombination -> layer_offset = N_reactions

    return shared_dydts[threadIdx.x];
}
