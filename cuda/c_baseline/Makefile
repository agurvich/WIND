LINEARALGEBRAFILES=$(wildcard linear_algebra/*.c)

GOLDODEFILE=../ode_system/ode_gold.c

RK2EXECUTABLE=../lib/rk2_gold.so
SIEEXECUTABLE=../lib/sie_gold.so

RK2OBJS=rk2/rk2_gold.o common/common_gold_RK2.o
SIEOBJS=sie/sie_gold.o common/common_gold_SIE.o


WINDCONFIG= -DADAPTIVE_TIMESTEP #-DDEBUGBLOCK=0 #-DLOUD

GOLDODEOBJS=$(GOLDODEFILE:%.c=%.o)
LINEARALGEBRAOBJS=$(LINEARALGEBRAFILES:%.c=%.o)

INCLUDE= -Ilinear_algebra -I. -I../ode_system -Iinclude -Icommon -I../include

#OPTFLAGS=-O3 #-g #
CFLAGS=-shared --compiler-options '-fPIC' $(OPTFLAGS)


make: $(GOLDODEOBJS) $(LINEARALGEBRAOBJS) $(RK2OBJS) $(SIEOBJS)
	nvcc $(CFLAGS) $(WINDCONFIG) $(INCLUDE) -o $(RK2EXECUTABLE) $(GOLDODEOBJS) $(RK2OBJS)
	nvcc $(CFLAGS) $(WINDCONFIG) $(INCLUDE) -o $(SIEEXECUTABLE) $(GOLDODEOBJS) $(LINEARALGEBRAOBJS) $(SIEOBJS)
common/common_gold_%.o:
	nvcc $(CFLAGS) $(WINDCONFIG) $(INCLUDE) -D$* -o $@ -c common/common_gold.c
%.o: %.c
	nvcc $(CFLAGS) $(WINDCONFIG) $(INCLUDE) -o $@ -c $<
preclean:
	rm -f ../ode_system/precompile_ode_gold.o
clean:
	rm -f $(RK2OBJS) $(SIEOBJS) $(GOLDODEOBJS) $(LINEARALGEBRAOBJS) $(RK2EXECUTABLE) $(SIEEXECUTABLE)
