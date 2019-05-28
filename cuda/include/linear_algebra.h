__global__ void gjeInvertMatrixBatched(
    float *, // d_matricess_flat,
    float *, // d_inversess_flat,
    int ,//Ndim,
    int );//Nbatch)

__device__ void gjeInvertMatrix(
    float *, // d_matrix
    float *, // d_inverse
    int, // Ndim
    float *); //shared_array
