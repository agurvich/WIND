## Library names
EXECUTABLE	:= ../lib/sie.so
ORDER2EXECUTABLE:= ../lib/sie2.so

## libraries to include
INCLUDE = -I../lib -I../include

BUILDDIR = $(shell pwd)

## compiler flags
NVCFLAGS = -g -G
make: 
	@## compile this guy
	nvcc $(CFLAGS) $(NVCFLAGS) -arch=sm_35 -ldl $(INCLUDE) -o debug debug.c;
rebuild:
	cd ..;make clean; make
	make
clean:
	rm debug
