# WIND
This is a repository.

Installation Instruction:

1) Clone the repository
2) cd WIND/cuda
2) mkdir lib 
3) mkdir RK2 
4) mkdir SIE
5) make
6) cd debug
7) make; ./debug

If 7) works then WIND is installed, hooray!

-- Advanced installation --
Installing the python frontend is a separate process...
1) install conda (https://docs.conda.io/en/latest/miniconda.html)
2) mkdir WIND/data
3) install required python modules:
  a) numpy 
  b) h5py
  c) matplotlib
  d) scipy
  e) pandas
  f) pip install memory-profiler
  g) psutil
4) mkdir ~/python; cd ~/python
5) ln -s ~/path/to/WIND wind and add ~/python to your PYTHONPATH environment variable
6) install chimes_driver
  a) chimes-driver
    I) cd ~/python; ln -s /path/to/chimes-driver chimes_driver
  b) chimes
    I) sundials static library not necessary,  nor do you have to build chimes at all
  c) chimes-data
7) clone abg_python (github.com/agurvich/abg_python) into ~/python (or add into your PYTHONPATH)
8) move device_memory_profile into $PATH variable
9) cd /path/to/wind/python; bash scaling_tests/single_run.sh

If 9) works then the python frontend is installed, hooray! Note that if 9) fails halfway through and creates a directory in WIND/data then you have to delete that directory otherwise the bash script will skip it, thinking it has no work to do. 



CA added this line to the README to try committing and pushing a code update.
