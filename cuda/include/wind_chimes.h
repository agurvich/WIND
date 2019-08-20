// declare the texture pointers in global memory 
ChimesFloat * wind_chimes_table_constant_rates;
cudaTextureObject_t wind_chimes_table_T_dependent_rates=0; // 1d layered texture


// define the actual structs...
extern struct wind_chimes_T_dependent_struct 
{ 
  int N_reactions[2]; 
  int **reactants; 
  int **products; 
  int H2_collis_dissoc_heating_reaction_index; 
  int H2_form_heating_reaction_index; 
  cudaTextureObject_t rates;
} wind_chimes_table_T_dependent;

