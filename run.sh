mkdir output
module reset
module load toolchain/foss/2023b
#module load toolchain/intel/2023b
#make clean
touch source/cuda_solver/solver/solver_iterator.cu
#touch source/cuda_solver/solver/iterator/solver_iterator_rk4.cu
rm main.o
#make SFML=FALSE FP32=FALSE TETM=FALSE CPU=TRUE COMPILER=g++
make SFML=FALSE FP32=TRUE TETM=FALSE CPU=TRUE COMPILER=g++

#export OMP_STACKSIZE=10g
export OMP_NUM_THREADS=$2
#$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=true

time ./main.o --L 200 200 --tmax 400 --gammaC 0.01 --gc 6E-6 --gr 12E-6 --initRandom 0.01 random --tmax 100 --outEvery 500000 --threads $OMP_NUM_THREADS --path output --N $1 $1 --output none --fftEvery 100000 --boundary periodic periodic
