TEXTUREFILE=texture_working.cu

## compiler flags
NVCFLAGS = -g -G -D_DEBUG
make: 
	@## compile this guy
	nvcc $(CFLAGS) $(NVCFLAGS) -arch=sm_35 -ldl -o texture $(TEXTUREFILE);
rebuild:
	cd ..;make clean; make
	make
clean:
	rm texture 
