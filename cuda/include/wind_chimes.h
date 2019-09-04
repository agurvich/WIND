extern "C" {
    #include "chimes_vars.h"
    #include "chimes_proto.h"
    void init_wind_chimes(struct globalVariables *);
}

// declare the wind_chimes structs as global variables
extern struct wind_chimes_constant_struct{ 
    int N_reactions[2];
    int H2_form_heating_reaction_index; 
    int * reactantss_transpose_flat; // 3xN_reactions_all list of reactants, flattened
    int * productss_transpose_flat; // 3xN_reactions_all list of products, flattened 
    ChimesFloat * rates; // 
} wind_chimes_table_constant;

extern struct wind_chimes_T_dependent_struct{ 
    int N_reactions[2];
    int H2_collis_dissoc_heating_reaction_index; 
    int H2_form_heating_reaction_index; 
    int * reactantss_transpose_flat; // 3xN_reactions_all list of reactants, flattened
    int * productss_transpose_flat; // 3xN_reactions_all list of products, flattened 
    cudaTextureObject_t rates; // N_Temperature texture with N_reactions_all layers
} wind_chimes_table_T_dependent;

extern struct wind_chimes_recombination_AB_struct{
    int N_reactions[2];
    int * reactantss_transpose_flat; // 3xN_reactions_all list of reactants, flattened
    int * productss_transpose_flat; // 3xN_reactions_all list of products, flattened 
    cudaTextureObject_t rates; // N_Temperature texture with 2*N_reactions_all layers, 
    //  A recomb are layers [0->N_reactions_all-1] 
    //  B recomb are layers [N_reactions_all -> 2*N_reactions_all-1]
} wind_chimes_table_recombination_AB;


extern struct wind_chimes_table_bins_struct{ 
  int N_Temperatures; 
  ChimesFloat *Temperatures; 
} wind_chimes_table_bins;

extern struct RHS_input_struct{
    struct wind_chimes_constant_struct * table_constant;
    struct wind_chimes_T_dependent_struct * table_T_dependent;
    struct wind_chimes_recombination_AB_struct * table_recombination_AB;

    struct wind_chimes_table_bins_struct * table_bins;
} * d_p_wind_chimes_RHS_input;
