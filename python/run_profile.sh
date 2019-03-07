module unload PrgEnv-cray darshan xalt
module load PrgEnv-pgi cudatoolkit

aprun -N 1 -n 1 profile_wrap.sh
