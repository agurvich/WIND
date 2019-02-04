#PBS -l nodes=1:ppn=1:xk
#PBS -l walltime=0:05:00
cd $PBS_O_WORKDIR
aprun -n1 ./hello > job.out
