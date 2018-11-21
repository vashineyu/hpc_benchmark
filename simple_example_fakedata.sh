#!/bin/bash

target_dir='record/experiment-1_node_2_gpu'
start_time=`date +%s`
CUDA_VISIBLE_DEVICES=1,2 mpirun -np 2 \
                                -H localhost:2 \
                                -bind-to none -map-by slot \
                                -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
                                python run_fakedata.py --result_dir ${target_dir} --test_mode 1
end_time=`date +%s`
runtime=$((end_time-start_time))
echo COND: ${target_dir}, Elapsed time: ${runtime} secs >> record/timer.txt
