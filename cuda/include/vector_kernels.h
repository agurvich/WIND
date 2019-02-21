
__global__ void overwriteVector(float *, float *);

__global__ void scaleVector(float *, float *);

__global__ void addVectors(float, float *, float, float *, float *);

__global__ void addArrayToBatchArrays(float **, float **, float, float, float);

__global__ void updateTimestep(float *, float *, float *, int *);

__global__ void checkError(float *, float *, int *);
