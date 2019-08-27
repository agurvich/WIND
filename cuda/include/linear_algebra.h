__global__ void gjeInvertMatrixBatched(
    WindFloat *, // d_matricess_flat,
    WindFloat *, // d_inversess_flat,
    int ,//Ndim,
    int );//Nbatch)

__device__ void gjeInvertMatrix(
    WindFloat *, // d_matrix
    WindFloat *, // d_inverse
    int, // Ndim
    WindFloat *); //shared_array
