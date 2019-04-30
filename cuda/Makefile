## WIND Configuration Options
ABSOLUTE_TOLERANCE=5e-3
RELATIVE_TOLERANCE=5e-3

WINDCONFIG= -DADAPTIVETIMESTEP #-DLOUD

WINDCONFIG+=-DABSOLUTE_TOLERANCE=$(ABSOLUTE_TOLERANCE)
WINDCONFIG+=-DRELATIVE_TOLERANCE=$(RELATIVE_TOLERANCE)

## Library names
EXECUTABLE	:= lib/sie.so
ORDER2EXECUTABLE:= lib/sie2.so



## Cuda source files (compiled with nvcc)
CUFILES		:= $(wildcard RK2/*.cu common_functions/*.cu ode_system.cu)

## objects
OBJS = $(CUFILES:%.cu=%.o)
SIEOBJS = SIE/SIE.o SIE/SIE2.o


## compiler flags
CFLAGS = --compiler-options '-fPIC' $(WINDCONFIG)
NVCFLAGS = -g -shared

## libraries to include
INCLUDE = -Iinclude -IRK2 -Icommon_functions##-I/sw/xe/magma/1.5.0/cnl5.2_gnu4.8.2/include
LIB = -lcublas -lcusolver

BUILDDIR = $(shell pwd)

make: $(CUFILES) $(OBJS) $(SIEOBJS)

SIE/SIE.o: SIE/SIE_harness.cu $(OBJS)
	@echo $@ specific foo $< $*
	@## compile this guy
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $@ -c $<;
	@## link the other files
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $(EXECUTABLE) $(OBJS) SIE/SIE.o;
SIE/SIE2.o: SIE/SIE_harness.cu $(OBJS)
	@echo $@ specific foo $< $*
	@## compile this guy
	nvcc $(NVCFLAGS) $(CFLAGS) -DORDER2 -arch=sm_35 $(INCLUDE) $(LIB) -o $@ -c $<;
	@## link the other files
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $(ORDER2EXECUTABLE) $(OBJS) SIE/SIE2.o;
%.o: %.cu
	@## compile this guy
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $@ -c $<;
clean: 
	rm $(OBJS)
	rm $(EXECUTABLE)
	rm $(ORDER2EXECUTABLE)
