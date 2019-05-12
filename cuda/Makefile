## WIND Configuration Options
ABSOLUTE_TOLERANCE=5e-3
RELATIVE_TOLERANCE=5e-3

WINDCONFIG= -DADAPTIVETIMESTEP #-DLOUD

WINDCONFIG+=-DABSOLUTE_TOLERANCE=$(ABSOLUTE_TOLERANCE)
WINDCONFIG+=-DRELATIVE_TOLERANCE=$(RELATIVE_TOLERANCE)

## Library names
SIEEXECUTABLE	:= lib/sie.so
RK2EXECUTABLE:= lib/rk2.so

## Cuda source files (compiled with nvcc)
CUFILES	:= linear_algebra/gauss_jordan_elimination.cu 
CUFILES+=$(wildcard common_functions/*.cu)

## which ode system to link to 
ODEFILE = ode_system/ode_system.cu ode_system/device_dydt.cu

## compiled object names
COMMONOBJS = $(CUFILES:%.cu=%.o)
ODEOBJ = $(ODEFILE:%.cu=%.o) 
SIEHOSTOBJS = SIE_host/harness_kernel.o
SIEOBJS = SIE/harness.o SIE/sie_kernel.o
RK2OBJS = RK2/harness.o RK2/rk2_kernel.o

## compiler flags to make shared object for python
CFLAGS = --compiler-options '-fPIC' $(WINDCONFIG)
NVCFLAGS = -shared -g -G -O0

## libraries to include
INCLUDE = -Iinclude -IRK2 -Icommon_functions -Iode_system -Ilinear_algebra
LIB = -lcublas -lcusolver

BUILDDIR = $(shell pwd)

make: $(c) $(COMMONOBJS) $(SIEOBJS) $(RK2OBJS) $(ODEOBJ)
	@#echo $(CUFILES)
	@## link common objs and ode obj to the 1st order semi-implicit solver 
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $(SIEEXECUTABLE) $(COMMONOBJS) $(ODEOBJ) $(SIEOBJS);
	@## link common objs and ode obj to the 2nd order explicit solver 
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $(RK2EXECUTABLE) $(COMMONOBJS) $(ODEOBJ) $(RK2OBJS);
	cd c_baseline;make
%.o: %.cu
	@#echo compiling $@ from $<
	@## compile this source file to object
	nvcc $(NVCFLAGS) $(CFLAGS) --device-c -c -arch=sm_35 $(INCLUDE) $(LIB) -o $@ $<;

clean: 
	rm $(COMMONOBJS) $(SIEOBJS) $(RK2OBJS) $(SIEEXECUTABLE) $(RK2EXECUTABLE)
	cd c_baseline;make clean
