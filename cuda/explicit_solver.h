const int THREAD_BLOCK_LIMIT = 1024;
__global__ void hello(int *, int *);
__global__ void integrate_euler(float, float, float *, float *, int, int); 
