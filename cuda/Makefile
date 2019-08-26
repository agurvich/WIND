## WIND Configuration Options
WINDCONFIG= -DADAPTIVE_TIMESTEP #-DDEBUGBLOCK=0 #-DLOUD
export WINDCONFIG

HDF5DIR=/opt/cray/hdf5/1.10.0/cray/83

CHIMESDIR=$(HOME)/src/CHIMES_repos/chimes


WINDCHIMESOBJS=CHIMES_TEMP/init_wind_chimes.o

CHIMESOBJS=$(wildcard $(CHIMESDIR)/test_runs/*.o)

CHIMESINCLUDE=-I$(CHIMESDIR) -I/home/abg6257/.local/include -I/software/hdf5/1.8.12-serial/include
CHIMESLIB = -L/home/abg6257/.local/lib -lsundials_cvode -lsundials_kinsol -lsundials_nvecserial -DH5_USE_16_API -L/software/hdf5/1.8.12-serial/lib -lhdf5



## OPTIMIZATION FLAGS
OPTFLAGS =  -O0 -g #
NVCOPTFLAGS = -G 
RK2OPTFLAGS=-Xptxas -O0
SIEOPTFLAGS=-Xptxas -O0
export OPTFLAGS

## Library names
SIEEXECUTABLE	:= lib/sie.so
RK2EXECUTABLE:= lib/rk2.so

## which ode system to link to 
ODEOBJ=device_dydt.o
HOSTODEOBJ=ode_system.o

SIEOBJS = SIE/harness.o SIE/kernel.o SIE/$(ODEOBJ) linear_algebra/gauss_jordan_elimination.o
RK2OBJS = RK2/harness.o RK2/kernel.o RK2/$(ODEOBJ)

## compiler flags to make shared object for python
CFLAGS = --compiler-options '-fPIC' $(WINDCONFIG) $(OPTFLAGS)
NVCFLAGS = -shared $(NVCOPTFLAGS)

## libraries to include
INCLUDE = -Iinclude -ISIE_host -ISIE -IRK2 -Iode_system -Ilinear_algebra $(CHIMESINCLUDE)
LIB = -lcublas -lcusolver $(CHIMESLIB)

BUILDDIR = $(shell pwd)

make: SIE RK2 baseline

### PHONY TARGETS
.PHONY: RK2
RK2: $(RK2OBJS) $(WINDCHIMESOBJS) $(CHIMESOBJS)
	@#echo making the phony $@ with $^
	@## link common objs and ode obj to the 2nd order explicit solver 
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $($@EXECUTABLE) $($@OBJS) $(WINDCHIMESOBJS) $(CHIMESOBJS);
.PHONY: SIE
SIE: $(SIEOBJS) $(WINDCHIMESOBJS) $(CHIMESOBJS) 
	@#echo making the phony $@ with $^
	@## link common objs and ode obj to the 1st order semi-implicit solver 
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=sm_35 $(INCLUDE) $(LIB) -o $($@EXECUTABLE) $($@OBJS) $(WINDCHIMESOBJS) $(CHIMESOBJS);
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
	rm -f $(RK2OBJS) $(SIEOBJS) $(WINDCHIMESOBJS)
	rm -f $(RK2EXECUTABLE) $(SIEEXECUTABLE)

