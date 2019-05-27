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

CFLAGS=-shared -fPIC -std=c99 -O3 #-g #

make: $(GOLDODEOBJS) $(LINEARALGEBRAOBJS) $(RK2OBJS) $(SIEOBJS)
	gcc $(CFLAGS) $(WINDCONFIG) $(INCLUDE) -o $(RK2EXECUTABLE) $(GOLDODEOBJS) $(RK2OBJS)
	gcc $(CFLAGS) $(WINDCONFIG) $(INCLUDE) -o $(SIEEXECUTABLE) $(GOLDODEOBJS) $(LINEARALGEBRAOBJS) $(SIEOBJS)
common/common_gold_%.o:
	gcc $(CFLAGS) $(WINDCONFIG) $(INCLUDE) -D$* -o $@ -c common/common_gold.c
%.o: %.c
	gcc $(CFLAGS) $(WINDCONFIG) $(INCLUDE) -o $@ -c $<
preclean:
	rm ../ode_system/precompile_ode_gold.o
clean:
	rm $(RK2OBJS) $(SIEOBJS) $(GOLDODEOBJS) $(LINEARALGEBRAOBJS) $(RK2EXECUTABLE) $(SIEEXECUTABLE)
