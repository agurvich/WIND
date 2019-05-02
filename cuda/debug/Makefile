## Library names
EXECUTABLE	:= ../lib/sie.so
ORDER2EXECUTABLE:= ../lib/sie2.so

## libraries to include
INCLUDE = -I../lib -I../include

make: 
	@## compile this guy
	nvcc -arch=sm_35 -ldl $(INCLUDE) -o debug debug.c;
clean:
	rm debug
