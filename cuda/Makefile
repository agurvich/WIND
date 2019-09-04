## WIND Configuration Options
WINDCONFIG= -DADAPTIVE_TIMESTEP -DLOUD #-DDEBUGBLOCK=0 #
export WINDCONFIG

## QUEST
#HDF5INCLUDE=/software/hdf5/1.8.12-serial/include
#HDF5LIB=/software/hdf5/1.8.12-serial/lib


## CUDACOMPUTE=sm_35 ## for quest and BW
CUDACOMPUTE=sm_60

## spark
HDF5INCLUDE=/usr/include/hdf5/serial
HDF5LIB=/usr/lib/x86_64-linux-gnu/hdf5/serial/

CHIMESDIR=$(HOME)/src/CHIMES_repos/chimes
CHIMESOBJS=$(wildcard $(CHIMESDIR)/test_runs/*.o)
CHIMESINCLUDE=-I$(CHIMESDIR) -I$(HOME)/.local/include -I$(HDF5INCLUDE)
CHIMESLIB = -L$(HOME)/.local/lib -lsundials_cvode -lsundials_kinsol -lsundials_nvecserial -DH5_USE_16_API -L$(HDF5LIB) -lhdf5

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
ODESOURCE := CHIMES_TEMP/device_dydt.cu
ODEOBJ := $(notdir $(ODESOURCE:.cu=.o))

## integrator objects
SIEOBJS = SIE/harness.o SIE/kernel.o SIE/$(ODEOBJ) linear_algebra/gauss_jordan_elimination.o
RK2OBJS = RK2/harness.o RK2/kernel.o RK2/$(ODEOBJ)

## wind chimes source files for chemistry-- should live in CHIMES eventually...
WINDCHIMESOBJS=CHIMES_TEMP/init_wind_chimes.o

## compiler flags to make shared object for python
CFLAGS = --compiler-options '-fPIC' $(WINDCONFIG) $(OPTFLAGS)
NVCFLAGS = -shared $(NVCOPTFLAGS)

## libraries to include
INCLUDE = -Iinclude -ISIE -IRK2 -Iode_system -Ilinear_algebra $(CHIMESINCLUDE)
LIB = $(CHIMESLIB) #-lcublas -lcusolver 

BUILDDIR = $(shell pwd)

make: SIE RK2 baseline

### PHONY TARGETS
.PHONY: RK2
RK2: $(WINDCHIMESOBJS) $(CHIMESOBJS) $(RK2OBJS)
	@#echo making the phony $@ with $^
	@## link common objs and ode obj to the 2nd order explicit solver 
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=$(CUDACOMPUTE) $(INCLUDE) $(LIB) -o $($@EXECUTABLE) $($@OBJS) $(WINDCHIMESOBJS) $(CHIMESOBJS);
.PHONY: SIE
SIE: $(WINDCHIMESOBJS) $(CHIMESOBJS) $(SIEOBJS) 
	@#echo making the phony $@ with $^
	@## link common objs and ode obj to the 1st order semi-implicit solver 
	nvcc $(NVCFLAGS) $(CFLAGS)  -arch=$(CUDACOMPUTE) $(INCLUDE) $(LIB) -o $($@EXECUTABLE) $($@OBJS) $(WINDCHIMESOBJS) $(CHIMESOBJS);
.PHONY: baseline
baseline:
	cd c_baseline;make
### NORMAL FILES
%.o: %.cu
	@#echo compiling $@ from $<
	@## compile this source file to object
	nvcc $(NVCFLAGS) $(CFLAGS) --device-c -c -arch=$(CUDACOMPUTE) $(INCLUDE) $(LIB) -o $@ $<;
### solver specific files 
%device_dydt.o: 
	@echo compiling $@ with $(*D) only $(*F) leftover-- compiling this device_dydt with whatever requested solver specific opt flags are 
	@# defined...
	nvcc $(NVCFLAGS) $($(*D)OPTFLAGS) $(CFLAGS) --device-c -c -arch=$(CUDACOMPUTE) -D$(*D) $(INCLUDE) $(LIB) -o $*device_dydt.o $(ODESOURCE);
SIE/%.o:
	@echo compiling $@ with SIE flags
	nvcc $(NVCFLAGS) $(SIEOPTFLAGS) $(CFLAGS) --device-c -c -arch=$(CUDACOMPUTE) -DSIE $(INCLUDE) $(LIB) -o $@ common/$*.cu
RK2/%.o:
	@#echo compiling $@ with RK2 flags
	nvcc $(NVCFLAGS) $(RK2OPTFLAGS) $(CFLAGS) --device-c -c -arch=$(CUDACOMPUTE) -DRK2 $(INCLUDE) $(LIB) -o $@ common/$*.cu
### Clean up
preclean:
	rm -f SIE/precompile_device_dydt.o 
	rm -f RK2/precompile_device_dydt.o
	cd c_baseline;make preclean
clean: 
	cd c_baseline;make clean
	rm -f $(RK2OBJS) $(SIEOBJS) $(WINDCHIMESOBJS)
	rm -f $(RK2EXECUTABLE) $(SIEEXECUTABLE)

