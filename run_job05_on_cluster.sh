#!/bin/bash
#PBS -P ENT107087
#PBS -N tf_horovod_four_node_multi-gpus
#PBS -l select=4:ncpus=24:ngpus=3:mpiprocs=3
#PBS -l walltime=48:00:00
#PBS -q gp16
#PBS -j oe
#PBS -M seanyu@aetherai.com
#PBS -m be 

module purge
module load singularity/2.5.2
module load cuda/9.1.85
module load anaconda3/5.1.10
module load openmpi/gcc/64/1.10.4
echo $PBS_O_WORKDIR

#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/aetherai2018/ # fix libcuda.so.1 error # already ln -s /pkg/cuda9.1/lib64/stubs/...
./init_environment.sh
source activate tf_keras
echo Start Running the Program

cd $PBS_O_WORKDIR
echo $PBS_NODEFILE
target_dir='record/experiment-4_node_3_gpu'
start_time=`date +%s`
mpirun -np 12 \
       -hostfile $PBS_NODEFILE \
       -bind-to none -map-by slot \
       -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
       python run.py --result_dir ${target_dir}

end_time=`date +%s`
runtime=$((end_time-start_time))
echo Condition ${target_dir} done, Elapsed time: ${runtime} secs

source deactivate
