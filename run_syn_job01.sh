#!/bin/bash
#PBS -P ENT107087
#PBS -N tfh_1-1
#PBS -l select=1:ncpus=10:ngpus=1:mpiprocs=1
#PBS -l walltime=24:00:00
#PBS -q gp4
#PBS -j oe
#PBS -M seanyu@aetherai.com
#PBS -m be 

module purge
module load singularity/2.5.2
module load cuda/9.1.85
module load anaconda3/5.1.10
module load openmpi/gcc/64/1.10.4
echo $PBS_O_WORKDIR

./init_environment.sh
source activate tf_keras
echo Start Running the Program

cd $PBS_O_WORKDIR
echo $PBS_NODEFILE
target_dir='record_syndata/experiment-1_node_1_gpu_10_cpu_v2'
start_time=`date +%s`
mpirun -np 1 \
       -hostfile $PBS_NODEFILE \
       -bind-to none -map-by slot \
       -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
       -mca pml ob1 -mca btl ^openib \
       python run_fakedata.py --result_dir ${target_dir}

end_time=`date +%s`
runtime=$((end_time-start_time))
echo Condition ${target_dir} done, Elapsed time: ${runtime} secs

source deactivate
