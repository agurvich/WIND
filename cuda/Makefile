# Add source files here
EXECUTABLE	:= lib/sie.so
ORDER2EXECUTABLE:= lib/sie2.so


BUILDDIR = $(shell pwd)

# Cuda source files (compiled with nvcc)
CUFILES		:= $(wildcard RK2/* common_functions/* ode_system.cu)

OBJS = $(CUFILES:%.cu=%.o)
SIEOBJS = SIE/SIE.o SIE/SIE2.o

CFLAGS = --compiler-options '-fPIC' -DADAPTIVETIMESTEP #-DLOUD 

NVCFLAGS = -g -shared

INCLUDE = -Iinclude -IRK2 -Icommon_functions##-I/sw/xe/magma/1.5.0/cnl5.2_gnu4.8.2/include

LIB = -lcublas -lcusolver

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
