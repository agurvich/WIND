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

__device__ void update_system(
    float * rates,
    float ** jacobian,
    struct wind_chimes_constant_struct * p_wind_chimes_table_constant,
    struct wind_chimes_T_dependent_struct * p_wind_chimes_table_T_dependent,
    struct wind_chimes_recombination_AB_struct * p_wind_chimes_table_recombination_AB,
    struct wind_chimes_table_bins_struct * p_wind_chimes_table_bins){

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
