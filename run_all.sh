#!/bin/bash
source activate tf_keras

target_dir='record/experiment-1_node_8_gpu'
start_time=`date +%s`
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -np 8 \
                                -H localhost:8 \
                                -bind-to none -map-by slot \
                                -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
                                python run.py --result_dir ${target_dir}
end_time=`date +%s`
runtime=$((end_time-start_time))
echo COND: ${target_dir}, Elapsed time: ${runtime} secs >> record/timer.txt


target_dir='record/experiment-1_node_4_gpu'
start_time=`date +%s`
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np 4 \
                                -H localhost:4 \
                                -bind-to none -map-by slot \
                                -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
                                python run.py --result_dir ${target_dir}
end_time=`date +%s`
runtime=$((end_time-start_time))
echo COND: ${target_dir}, Elapsed time: ${runtime} secs >> record/timer.txt


target_dir='record/experiment-1_node_2_gpu'
start_time=`date +%s`
CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 \
                                -H localhost:2 \
                                -bind-to none -map-by slot \
                                -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
                                python run.py --result_dir ${target_dir}
end_time=`date +%s`
runtime=$((end_time-start_time))
echo COND: ${target_dir}, Elapsed time: ${runtime} secs >> record/timer.txt

target_dir='record/experiment-1_node_1_gpu'
start_time=`date +%s`
CUDA_VISIBLE_DEVICES=7 mpirun -np 1 \
                                -H localhost:1 \
                                -bind-to none -map-by slot \
                                -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
                                python run.py --result_dir ${target_dir}
end_time=`date +%s`
runtime=$((end_time-start_time))
echo COND: ${target_dir}, Elapsed time: ${runtime} secs >> record/timer.txt
