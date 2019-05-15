## Library names
EXECUTABLE	:= ../lib/rk2_gold.so
DEBUGFILE=debug.c
## libraries to include
INCLUDE = -I../lib -I../include

BUILDDIR = $(shell pwd)

## compiler flags
NVCFLAGS = -g -G
make: 
	@## compile this guy
	nvcc $(CFLAGS) $(NVCFLAGS) -arch=sm_35 -ldl $(INCLUDE) -o debug $(DEBUGFILE);
rebuild:
	cd ..;make clean; make
	make
clean:
	rm debug
