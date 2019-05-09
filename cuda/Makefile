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
CUFILES	:=  linear_algebra/gauss_jordan_elimination.cu
CUFILES+=$(wildcard common_functions/*.cu)

## which ode system to link to 
ODEFILE = ode_system.cu

## compiled object names
COMMONOBJS = $(CUFILES:%.cu=%.o)
ODEOBJ = $(ODEFILE:%.cu=%.o) 
SIEOBJS = SIE/harness_kernel.o
RK2OBJS = RK2/harness.o RK2/kernel.o RK2/device_dydt.o

## compiler flags to make shared object for python
CFLAGS = --compiler-options '-fPIC' $(WINDCONFIG)
NVCFLAGS = -shared #-g -G -O0

## libraries to include
INCLUDE = -Iinclude -IRK2 -Icommon_functions -Ilinear_algebra##-I/sw/xe/magma/1.5.0/cnl5.2_gnu4.8.2/include
LIB = -lcublas -lcusolver

BUILDDIR = $(shell pwd)

make: $(c) $(COMMONOBJS) $(SIEOBJS) $(RK2OBJS) $(ODEOBJ)
	@#echo $(CUFILES)
	@## link common objs and ode obj to the 1st order semi-implicit solver 
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $(SIEEXECUTABLE) $(COMMONOBJS) $(ODEOBJ) $(SIEOBJS);
	@## link common objs and ode obj to the 2nd order explicit solver 
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $(RK2EXECUTABLE) $(COMMONOBJS) $(ODEOBJ) $(RK2OBJS);

%.o: %.cu
	@#echo compiling $@ from $<
	@## compile this source file to object
	nvcc $(NVCFLAGS) $(CFLAGS) --device-c -c -arch=sm_35 $(INCLUDE) $(LIB) -o $@ $<;

clean: 
	rm $(COMMONOBJS)
	rm $(SIEOBJS)
	rm $(RK2OBJS)
	rm $(SIEEXECUTABLE)
	rm $(RK2EXECUTABLE)
	rm *.*o 
	rm lib/*.*o
