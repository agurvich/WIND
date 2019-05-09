## WIND Configuration Options
ABSOLUTE_TOLERANCE=5e-3
RELATIVE_TOLERANCE=5e-3

WINDCONFIG= #-DADAPTIVETIMESTEP #-DLOUD

WINDCONFIG+=-DABSOLUTE_TOLERANCE=$(ABSOLUTE_TOLERANCE)
WINDCONFIG+=-DRELATIVE_TOLERANCE=$(RELATIVE_TOLERANCE)

## Library names
EXECUTABLE	:= lib/sie.so
ORDER2EXECUTABLE:= lib/sie2.so

## Cuda source files (compiled with nvcc)
CUFILES	:= RK2/harness.cu RK2/kernel.cu 
CUFILES+=$(wildcard common_functions/*.cu)

## ode system to link to 
ODEFILE = ode_system.cu
ODEOBJ =$(ODEFILE:%.cu=%.o) 

## objects
OBJS = $(CUFILES:%.cu=%.o)
SIEOBJS = SIE/SIE.o SIE/SIE2.o

## compiler flags
CFLAGS = --compiler-options '-fPIC' $(WINDCONFIG)
NVCFLAGS = -shared #-g -G -O0

## libraries to include
INCLUDE = -Iinclude -IRK2 -Icommon_functions##-I/sw/xe/magma/1.5.0/cnl5.2_gnu4.8.2/include
LIB = -lcublas -lcusolver

BUILDDIR = $(shell pwd)

make: $(CUFILES) $(OBJS) $(SIEOBJS) $(ODEFILE)
	@## compile the ODE equations
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $(ODEOBJ) -c $(ODEFILE);
	@## link to the first order solver 
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $(EXECUTABLE) $(OBJS) $(ODEOBJ) SIE/SIE.o;
	@## link the the second order solver
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $(ORDER2EXECUTABLE) $(OBJS) $(ODEOBJ) SIE/SIE2.o;

SIE/SIE.o: SIE/SIE_harness.cu
	@echo $@ specific foo $< $*
	@## compile this guy
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $@ -c $<;

SIE/SIE2.o: SIE/SIE_harness.cu
	@echo $@ specific foo $< $*
	@## compile this guy
	nvcc $(NVCFLAGS) $(CFLAGS) -DMIDPOINT -arch=sm_35 $(INCLUDE) $(LIB) -o $@ -c $<;

%.o: %.cu
	@## compile this guy
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $@ -c $<;
clean: 
	rm $(OBJS)
	rm $(SIEOBJS)
	rm $(EXECUTABLE)
	rm $(ORDER2EXECUTABLE)
	rm *.*o 
	rm lib/*.*o
