## WIND Configuration Options
WINDCONFIG= -DADAPTIVE_TIMESTEP #-DDEBUGBLOCK=0 #-DLOUD
export WINDCONFIG

## OPTIMIZATION FLAGS
OPTFLAGS =  -O3 #-g #
NVCOPTFLAGS = #-G 
RK2OPTFLAGS=-Xptxas -O3
SIEOPTFLAGS=-Xptxas -O3
export OPTFLAGS

## Library names
SIEEXECUTABLE	:= lib/sie.so
SIEHOSTEXECUTABLE   := lib/sie_host.so
RK2EXECUTABLE:= lib/rk2.so

## which ode system to link to 
ODEOBJ=device_dydt.o
HOSTODEOBJ=ode_system.o

SIEHOSTOBJS = SIE_host/harness_kernel.o linear_algebra/gauss_jordan_elimination.o
SIEHOSTOBJS +=SIE_host/utils.o SIE_host/vector_kernels.o SIE_host/cuda_utils.o ode_system/$(HOSTODEOBJ)

SIEOBJS = SIE/harness.o SIE/kernel.o linear_algebra/gauss_jordan_elimination.o SIE/$(ODEOBJ)
RK2OBJS = RK2/harness.o RK2/kernel.o RK2/$(ODEOBJ)

## compiler flags to make shared object for python
CFLAGS = --compiler-options '-fPIC' $(WINDCONFIG) $(OPTFLAGS)
NVCFLAGS = -shared $(NVCOPTFLAGS)

## libraries to include
INCLUDE = -Iinclude -ISIE_host -ISIE -IRK2 -Iode_system -Ilinear_algebra
LIB = -lcublas -lcusolver

BUILDDIR = $(shell pwd)

## don't make SIEhost... no point
#SIEHOST
make: SIE RK2 baseline SIEHOST

### PHONY TARGETS
.PHONY: RK2
RK2: $(RK2OBJS)
	@#echo making the phony $@ with $^
	@## link common objs and ode obj to the 2nd order explicit solver 
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $($@EXECUTABLE) $($@OBJS);
.PHONY: SIE
SIE: $(SIEOBJS)
	@#echo making the phony $@ with $^
	@## link common objs and ode obj to the 1st order semi-implicit solver 
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $($@EXECUTABLE) $($@OBJS);
.PHONY: SIEHOST
SIEHOST: $(SIEHOSTOBJS)
	@## link common objs and ode obj to the 1st order semi-implicit that is host-locked
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $($@EXECUTABLE) $($@OBJS);
.PHONY: baseline
baseline:
	cd c_baseline;make
### NORMAL FILES
%.o: %.cu
	@#echo compiling $@ from $<
	@## compile this source file to object
	nvcc $(NVCFLAGS) $(CFLAGS) --device-c -c -arch=sm_35 $(INCLUDE) $(LIB) -o $@ $<;
### solver specific files
%device_dydt.o:
	@#echo compiling $@ with $(*D) only $(*F) leftover
	nvcc $(NVCFLAGS) $($(*D)OPTFLAGS) $(CFLAGS) --device-c -c -arch=sm_35 $(INCLUDE) $(LIB) -o $*device_dydt.o ode_system/$(*F)device_dydt.cu;
SIE/%.o:
	@#echo compiling $@ with SIE flags
	nvcc $(NVCFLAGS) $(SIEOPTFLAGS) $(CFLAGS) --device-c -c -arch=sm_35 -DSIE $(INCLUDE) $(LIB) -o $@ common/$*.cu
RK2/%.o:
	@#echo compiling $@ with RK2 flags
	nvcc $(NVCFLAGS) $(RK2OPTFLAGS) $(CFLAGS) --device-c -c -arch=sm_35 -DRK2 $(INCLUDE) $(LIB) -o $@ common/$*.cu
### Clean up
preclean:
	rm -f SIE/precompile_device_dydt.o 
	rm -f RK2/precompile_device_dydt.o
	rm -f ode_system/precompile_ode_system.o
	cd c_baseline;make preclean
clean: 
	cd c_baseline;make clean
	rm -f $(RK2OBJS) $(SIEOBJS) $(SIEHOSTOBJS)
	rm -f $(RK2EXECUTABLE) $(SIEEXECUTABLE) $(SIEHOSTEXECUTABLE)

