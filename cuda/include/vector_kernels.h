#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS_PER_GRID 65535

__global__ void overwriteVector(float *, float *, int, int);

__global__ void scaleVector(float *, float, int, int);

__global__ void addVectors(float, float *, float, float *, float *, int, int);

__global__ void checkError(float *, float *, int *, int, int, float, float);

__global__ void addArrayToBatchArrays(float **, float **, float, float, float, int, int);

__global__ void updateTimestep(float *, float *, float *, int *);
