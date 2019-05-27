## WIND Configuration Options
WINDCONFIG= -DADAPTIVE_TIMESTEP -DDEBUGBLOCK=0 #-DLOUD
export WINDCONFIG

## Library names
SIEEXECUTABLE	:= lib/sie.so
SIEHOSTEXECUTABLE   := lib/sie_host.so
RK2EXECUTABLE:= lib/rk2.so

## Cuda source files (compiled with nvcc)
CUFILES	:= linear_algebra/gauss_jordan_elimination.cu 
CUFILES+= common_functions/utils.cu common_functions/vector_kernels.cu common_functions/cuda_utils.cu

## which ode system to link to 
ODEFILE = ode_system/ode_system.cu ode_system/device_dydt.cu
GOLDODEFILE= ode_system/ode_gold.c

## compiled object names
COMMONOBJS = $(CUFILES:%.cu=%.o)
ODEOBJS = $(ODEFILE:%.cu=%.o) 
SIEHOSTOBJS = SIE_host/harness_kernel.o
SIEOBJS = harnesses/harness_SIE.o kernels/kernel_SIE.o
RK2OBJS = harnesses/harness_RK2.o kernels/kernel_RK2.o

## compiler flags to make shared object for python
CFLAGS = --compiler-options '-fPIC' $(WINDCONFIG)
NVCFLAGS = -shared -g -G #-O3 #

## libraries to include
INCLUDE = -Iinclude -Icommon_functions -Iode_system -Ilinear_algebra
LIB = -lcublas -lcusolver

BUILDDIR = $(shell pwd)

#export PRECOMPILE
make: $(c) $(COMMONOBJS) $(SIEOBJS) $(RK2OBJS) $(SIEHOSTOBJS) $(ODEOBJS) baseline
	@#echo $(ODEOBJS)
	@## link common objs and ode obj to the 1st order semi-implicit solver 
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $(SIEEXECUTABLE) $(COMMONOBJS) $(ODEOBJS) $(SIEOBJS);
	@## link common objs and ode obj to the 1st order semi-implicit that is host-locked
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $(SIEHOSTEXECUTABLE) $(COMMONOBJS) $(ODEOBJS) $(SIEHOSTOBJS);
	@## link common objs and ode obj to the 2nd order explicit solver 
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $(RK2EXECUTABLE) $(COMMONOBJS) $(ODEOBJS) $(RK2OBJS);
baseline:
	cd c_baseline;make
%.o: %.cu
	@#echo compiling $@ from $<
	@## compile this source file to object
	nvcc $(NVCFLAGS) $(CFLAGS) --device-c -c -arch=sm_35 $(INCLUDE) $(LIB) -o $@ $<;
kernels/kernel_%.o:
	@#echo compiling $@ with $*
	nvcc $(NVCFLAGS) $(CFLAGS) --device-c -c -arch=sm_35 -D$* $(INCLUDE) $(LIB) -o $@ kernels/kernel.cu
harnesses/harness_%.o:
	@#echo compiling $@ with $*
	nvcc $(NVCFLAGS) $(CFLAGS) --device-c -c -arch=sm_35 -D$* $(INCLUDE) $(LIB) -o $@ harnesses/harness.cu
preclean:
	rm ode_system/precompile_ode_system.o ode_system/precompile_device_dydt.o
	cd c_baseline;make preclean
clean: 
	rm $(COMMONOBJS) $(SIEOBJS) $(RK2OBJS) $(SIEHOSTOBJS) $(ODEOBJS) $(SIEEXECUTABLE) $(RK2EXECUTABLE) $(SIEHOSTEXECUTABLE)
	cd c_baseline;make clean
