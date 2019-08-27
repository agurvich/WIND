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

    int this_react;
    // calculate the rate itself by multiplying by reactant abundances and 
    //  factors of nH
    for (int reactant_i=0; reactant_i<N_reactants; reactant_i++){
        this_react = reactantss[tid + reactant_i*N_reactions];
        if (this_react>0) this_rate_coeff*=nH*shared_equations[this_react];
        this_rate_coeff/=nH; // want Nreacts-1 many factors of nH
    }


    // update creation rates in shared_dydts
    for (int reactant_i=0; reactant_i<N_reactants; reactant_i++){
        this_react = reactantss[tid + reactant_i*N_reactions];
        if (this_react >0){
            atomicAdd((float *) &shared_dydts[this_react],(float) -this_rate_coeff);
// update Jacobian
#ifdef SIE
            for (int prod_i=0; prod_i<N_products; prod_i++){
                this_prod = productss[tid + prod_i*N_reactions];
                if (this_prod >0){ 
                    atomicAdd(
                        &Jacobians[f(this_react,this_prod)],
                        -this_rate_coeff/shared_equations[this_prod]);
                    atomicAdd(
                        &Jacobians[f(this_prod,this_react)],
                        this_rate_coeff/shared_equations[this_react]);
                } // if this_prod >0
            } // for prod in prods
#endif
        }// if this_react > 0 
    }// for react in reacts

    int this_prod;
    // update destruction rates in shared_dydts
    for (int prod_i=0; prod_i<N_products; prod_i++){
        this_prod = productss[tid + prod_i*N_reactions];
        if (this_prod >0) {
            atomicAdd((float *) &shared_dydts[this_prod],(float) this_rate_coeff);
        }
    }
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
    for (int rxn_i=0; rxn_i < N_reactions/blockDim.x; rxn_i++){
        // do blockDim.x many reactions simultaneously
        tid = rxn_i*blockDim.x + threadIdx.x;

        // do we have more reactions to solve?
        if (tid < N_reactions){
            // read rate directly from array
            this_rate_coeff = rate_coeffs[tid];

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
    for (int rxn_i=0; rxn_i < N_reactions/blockDim.x; rxn_i++){
        // do blockDim.x many reactions simultaneously
        tid = rxn_i*blockDim.x + threadIdx.x;

        // do we have more reactions to solve?
        if (tid < N_reactions){
            // read off the rate coefficient from the texture
            this_rate_coeff = pow(10.0,
                (ChimesFloat) tex1DLayered<float>(
                    rate_coeff_tex,T_tex_coord,layer_offset+tid));

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

    // cast input struct to the correct format
    struct RHS_input_struct * p_RHS_input = (struct RHS_input_struct *) RHS_input;

    // unpack the RHS_input struct
    struct wind_chimes_constant_struct * p_wind_chimes_table_constant = p_RHS_input->table_constant;
    struct wind_chimes_T_dependent_struct * p_wind_chimes_table_T_dependent = p_RHS_input->table_T_dependent;
    struct wind_chimes_recombination_AB_struct * p_wind_chimes_table_recombination_AB = p_RHS_input->table_recombination_AB;
    struct wind_chimes_table_bins_struct * p_wind_chimes_table_bins = p_RHS_input->table_bins;

    // constantss_flat = [T0,nH0,T1,nH1,T2,nH2...] and constants = &constantss_flat[NUM_CONST*blockIdx.x]
    WindFloat Temperature = constants[0];
    WindFloat nH = constants[1]; 

    WindFloat TMOL = 100.0;
    
/* ------- chimes_table_constant ------- */
    loop_over_reactions_constant(
        p_wind_chimes_table_constant->N_reactions[Temperature<TMOL],
        p_wind_chimes_table_constant->reactantss_transpose_flat,2,
        p_wind_chimes_table_constant->productss_transpose_flat,3,
        p_wind_chimes_table_constant->rates,
        nH,
        shared_equations,
        shared_dydts,
        Jacobians);

/* ------- chimes_table_T_dependent ------- */
    float T_tex_coord = determine_interpolation_range(
        Temperature,
        p_wind_chimes_table_bins->Temperatures,
        p_wind_chimes_table_bins->N_Temperatures);
    
    loop_over_reactions_T_dependent(
        p_wind_chimes_table_T_dependent->N_reactions[Temperature<TMOL],
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
        p_wind_chimes_table_recombination_AB->N_reactions[Temperature<TMOL],
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
