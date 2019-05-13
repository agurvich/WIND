COMMONFILES=common_gold.c
ODEFILE=ode_gold.c
RK2FILE=rk2_gold.c
RK2EXECUTABLE=rk2_gold.so

CONFIG= -DADAPTIVE_TIMESTEP #-DLOUD
CONFIG+= -DABSOLUTE_TOLERANCE=5e-3
CONFIG+= -DRELATIVE_TOLERANCE=5e-3

COMMONOBJS=$(COMMONFILES:%.c=%.o)
ODEOBJ=$(ODEFILE:%.c=%.o)

CFLAGS=-shared -fPIC -g
make: $(COMMONOBJS) $(ODEOBJ)
	cc $(CFLAGS) $(CONFIG) -o $(RK2EXECUTABLE) $(COMMONOBJS) $(ODEOBJ) $(RK2FILE)
%.o: %.c
	cc $(CFLAGS) $(CONFIG) -o $@ -c $<
clean:
	rm $(COMMONOBJS) $(ODEOBJ) $(RK2EXECUTABLE)