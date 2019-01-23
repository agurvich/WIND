# Add source files here
EXECUTABLE	:= lib/wind.so

# Cuda source files (compiled with nvcc)
CUFILES		:= harness.cu kernel.cu SIE_harness_kernels.cu

CFLAGS = --compiler-options '-fPIC' -shared

INCLUDE = ##-I/sw/xe/magma/1.5.0/cnl5.2_gnu4.8.2/include

LIB = -lcublas -lcusolver

make: $(CUFILES)
	nvcc $(CFLAGS) -arch=sm_35 $(INCLUDE) $(LIB) -o $(EXECUTABLE) $(CUFILES) $(CUDEPS);
clean: 
	rm lib/*.so

