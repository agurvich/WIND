
// prints each element of an array of floats
void printFArray(float *, int);

// fills the value of a matrix of size N with the identity
void setIdentityMatrix(float *, int);

// allocates a flat and 2d array for a matrix on the device
float ** initializeDeviceMatrix(float *, float **, int, int);
