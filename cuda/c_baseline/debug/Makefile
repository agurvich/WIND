## Library names
EXECUTABLE	:= ../lib/rk2_gold.so
DEBUGFILE=debug.c
## libraries to include
INCLUDE = -I../lib -I../include

BUILDDIR = $(shell pwd)

## compiler flags
CFLAGS = -g
make: 
	@## compile this guy
	nvcc $(CFLAGS) $(NVCFLAGS) -ldl $(INCLUDE) -o debug $(DEBUGFILE);
rebuild:
	cd ..;make clean; make
	make
clean:
	rm debug
