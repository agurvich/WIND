// declare the texture pointers in global memory 
float * wind_chimes_table_constant;

extern struct wind_chimes_T_dependent_struct{
    int N_reactions[2];
    int H2_collis_dissoc_heating_reaction_index; 
    int H2_form_heating_reaction_index; 
    int * reactantss_transpose_flat; // 3xN_reactions_all list of reactants, flattened
    int * productss_transpose_flat; // 3xN_reactions_all list of products, flattened 
    void * rates; // void -> cudaTextureObject_t or ChimesFloat, cast in _device code
} wind_chimes_table_T_dependent wind_chimes_table_AB_recombination;

// define a table
extern struct Table_struct{

} table;
