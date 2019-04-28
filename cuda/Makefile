# Add source files here
EXECUTABLE	:= lib/wind.so

# Cuda source files (compiled with nvcc)
CUFILES		:= RK2/* SIE/* BDF2/* common_functions/* ode_system.cu

CFLAGS = --compiler-options '-fPIC' -DADAPTIVETIMESTEP #-DLOUD 

NVCFLAGS = -g -shared

INCLUDE = -Iinclude ##-I/sw/xe/magma/1.5.0/cnl5.2_gnu4.8.2/include

LIB = -lcublas -lcusolver

make: $(CUFILES)
	nvcc $(NVCFLAGS) $(CFLAGS) -arch=sm_35 $(INCLUDE) $(LIB) -o $(EXECUTABLE) $(CUFILES) $(CUDEPS);
clean: 
	rm lib/*.so

