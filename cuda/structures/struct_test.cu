#include <stdio.h>

// define a table
extern struct Table_struct{
    int N_reactions[2];
    int H2_collis_dissoc_heating_reaction_index; 
    int H2_form_heating_reaction_index; 
    int * reactantss_transpose_flat; // 3xN_reactions_all list of reactants, flattened
    int * productss_transpose_flat; // 3xN_reactions_all list of products, flattened 
    float * rates;
} table;

__global__ void readStruct(int x, struct Table_struct * p_table){
    struct Table_struct this_table = *p_table;
    int N_reactions_all = this_table.N_reactions[1];
    printf("%d Nreactions\n",N_reactions_all);
    // loop over all the equations in parallel, with whatever threads we were given..
    int idx;
    for (int rxn_loop_i=0; rxn_loop_i < (N_reactions_all/blockDim.x+1); rxn_loop_i++){
        idx = threadIdx.x+rxn_loop_i*blockDim.x;
        if (idx < N_reactions_all){
            float this_rate_coeff = this_table.rates[idx];
            printf("%.2f ",this_rate_coeff);
            for (int reactant_i=0; reactant_i < 3; reactant_i++){ // 3 reactants in the reactants list
                printf("n_{%d} ",
                    this_table.reactantss_transpose_flat[idx+reactant_i*N_reactions_all]);
            }// for react_i
        }
    }// for rxn_loop_i
}

// create a new global table
struct Table_struct table;

void initialize_table(
    struct Table_struct * p_this_table,
    int * N_reactions,
    int * reactantss_transpose_flat,
    int * productss_transpose_flat,
    int H2_collis_dissoc_heating_reaction_index,
    int H2_form_heating_reaction_index,
    float * rates){

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

int main(){
    // initialize fake host data
    int H2_collis_dissoc_heating_reaction_index=0;  // random data
    int H2_form_heating_reaction_index=0; 

    int N_reactions[2] = {1,1};
    int reactantss_transpose_flat[3] = {1,0,-1};
    int productss_transpose_flat[3] = {1,2,-1};
    float rates[1] = {0.5};

    int num_species = 5;


    // add N_reactions
    initialize_table(
        &table,
        N_reactions,
        reactantss_transpose_flat,
        productss_transpose_flat,
        H2_collis_dissoc_heating_reaction_index,
        H2_form_heating_reaction_index,
        rates); 
    printf(" n reactions is %d %d \n",table.N_reactions[0],table.N_reactions[1]);

    // allocate a table on the device, copy over the host structure over
    struct Table_struct * d_p_table;
    cudaMalloc(&d_p_table,sizeof(Table_struct));
    cudaMemcpy(d_p_table,&table,sizeof(Table_struct),cudaMemcpyHostToDevice);

    printf("hello world from the host\n");
    readStruct<<<1,num_species>>>(1,d_p_table);
    cudaDeviceSynchronize();
    printf("good night from the host\n");
    
}
