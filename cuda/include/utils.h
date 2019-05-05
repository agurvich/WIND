
// prints each element of an array of floats
void printFArray(float *, int);

// fills the value of a matrix of size N with the identity
void setIdentityMatrix(float *, int);

// allocates a flat and 2d array for a matrix on the device
float ** initializeDeviceMatrix(float *, float **, int, int);

// just a function call to set a breakpoint on for cuda-gdb
void GDBbreakpoint();

// just a function to create and destroy a cublas handle
//  to avoid timing interference. 
void initializeCublasExternally();
