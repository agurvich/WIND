## WIND Configuration Options
ABSOLUTE_TOLERANCE=5e-3
RELATIVE_TOLERANCE=5e-3

WINDCONFIG= -DADAPTIVE_TIMESTEP #-DDEBUGBLOCK=0 #-DLOUD

WINDCONFIG+=-DABSOLUTE_TOLERANCE=$(ABSOLUTE_TOLERANCE)
WINDCONFIG+=-DRELATIVE_TOLERANCE=$(RELATIVE_TOLERANCE)

## Library names
SIEEXECUTABLE	:= lib/sie.so
SIEHOSTEXECUTABLE   := lib/sie_host.so
RK2EXECUTABLE:= lib/rk2.so

## Cuda source files (compiled with nvcc)
CUFILES	:= linear_algebra/gauss_jordan_elimination.cu 
CUFILES+=$(wildcard common_functions/*.cu)

## which ode system to link to 
ODEFILE = ode_system/ode_system.cu ode_system/device_dydt.cu
GOLDODEFILE= ode_system/ode_gold.c

## compiled object names
COMMONOBJS = $(CUFILES:%.cu=%.o)
ODEOBJS = $(ODEFILE:%.cu=%.o) 
SIEHOSTOBJS = SIE_host/harness_kernel.o
SIEOBJS = SIE/harness.o SIE/sie_kernel.o
RK2OBJS = RK2/harness.o RK2/rk2_kernel.o

## compiler flags to make shared object for python
CFLAGS = --compiler-options '-fPIC' $(WINDCONFIG)
NVCFLAGS = -shared -g -G

## libraries to include
INCLUDE = -Iinclude -IRK2 -Icommon_functions -Iode_system -Ilinear_algebra
LIB = -lcublas -lcusolver

BUILDDIR = $(shell pwd)

#export PRECOMPILE
make: $(c) $(COMMONOBJS) $(SIEOBJS) $(RK2OBJS) $(SIEHOSTOBJS) $(ODEOBJS)
	@echo $(ODEOBJS)
	@## link common objs and ode obj to the 1st order semi-implicit solver 
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $(SIEEXECUTABLE) $(COMMONOBJS) $(ODEOBJS) $(SIEOBJS);
	@## link common objs and ode obj to the 1st order semi-implicit that is host-locked
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $(SIEHOSTEXECUTABLE) $(COMMONOBJS) $(ODEOBJS) $(SIEHOSTOBJS);
	@## link common objs and ode obj to the 2nd order explicit solver 
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $(RK2EXECUTABLE) $(COMMONOBJS) $(ODEOBJS) $(RK2OBJS);
	cd c_baseline;make
%.o: %.cu
	@#echo compiling $@ from $<
	@## compile this source file to object
	nvcc $(NVCFLAGS) $(CFLAGS) --device-c -c -arch=sm_35 $(INCLUDE) $(LIB) -o $@ $<;

preclean:
	rm ode_system/precompile_ode_system.o ode_system/precompile_device_dydt.o
	cd c_baseline;make preclean
clean: 
	rm $(COMMONOBJS) $(SIEOBJS) $(RK2OBJS) $(SIEHOSTOBJS) $(ODEOBJS) $(SIEEXECUTABLE) $(RK2EXECUTABLE) $(SIEHOSTEXECUTABLE)
	cd c_baseline;make clean
