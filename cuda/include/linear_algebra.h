#include <cublas_v2.h>

__global__ void gjeInvertMatrixBatched(
    float *, // d_matricess_flat,
    float *, // d_inversess_flat,
    int ,//Ndim,
    int );//Nbatch)
