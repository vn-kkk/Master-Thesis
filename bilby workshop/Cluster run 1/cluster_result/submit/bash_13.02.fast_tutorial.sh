#!/usr/bin/env bash

# 13-02-fast_tutorial_data0_0_generation
# PARENTS 
# CHILDREN 13-02-fast_tutorial_data0_0_analysis_H1L1_par0 13-02-fast_tutorial_data0_0_analysis_H1L1_par1
if [[ "13-02-fast_tutorial_data0_0_generation" == *"$1"* ]]; then
    echo "Running: /cluster/users/venkatek/.conda/envs/bilby/bin/bilby_pipe_generation outdir_13.02.fast_tutorial/13.02.fast_tutorial_config_complete.ini --label 13-02-fast_tutorial_data0_0_generation --idx 0 --trigger-time 0 --injection-file outdir_13.02.fast_tutorial/data/13.02.fast_tutorial_injection_file.dat --outdir outdir_13.02.fast_tutorial"
    /cluster/users/venkatek/.conda/envs/bilby/bin/bilby_pipe_generation outdir_13.02.fast_tutorial/13.02.fast_tutorial_config_complete.ini --label 13-02-fast_tutorial_data0_0_generation --idx 0 --trigger-time 0 --injection-file outdir_13.02.fast_tutorial/data/13.02.fast_tutorial_injection_file.dat --outdir outdir_13.02.fast_tutorial
fi

# 13-02-fast_tutorial_data0_0_analysis_H1L1_par0
# PARENTS 13-02-fast_tutorial_data0_0_generation
# CHILDREN 13-02-fast_tutorial_data0_0_analysis_H1L1_merge
if [[ "13-02-fast_tutorial_data0_0_analysis_H1L1_par0" == *"$1"* ]]; then
    echo "Running: /cluster/users/venkatek/.conda/envs/bilby/bin/bilby_pipe_analysis outdir_13.02.fast_tutorial/13.02.fast_tutorial_config_complete.ini --outdir outdir_13.02.fast_tutorial --detectors H1 --detectors L1 --label 13-02-fast_tutorial_data0_0_analysis_H1L1_par0 --data-dump-file outdir_13.02.fast_tutorial/data/13-02-fast_tutorial_data0_0_generation_data_dump.pickle --sampler dynesty"
    /cluster/users/venkatek/.conda/envs/bilby/bin/bilby_pipe_analysis outdir_13.02.fast_tutorial/13.02.fast_tutorial_config_complete.ini --outdir outdir_13.02.fast_tutorial --detectors H1 --detectors L1 --label 13-02-fast_tutorial_data0_0_analysis_H1L1_par0 --data-dump-file outdir_13.02.fast_tutorial/data/13-02-fast_tutorial_data0_0_generation_data_dump.pickle --sampler dynesty
fi

# 13-02-fast_tutorial_data0_0_analysis_H1L1_par1
# PARENTS 13-02-fast_tutorial_data0_0_generation
# CHILDREN 13-02-fast_tutorial_data0_0_analysis_H1L1_merge
if [[ "13-02-fast_tutorial_data0_0_analysis_H1L1_par1" == *"$1"* ]]; then
    echo "Running: /cluster/users/venkatek/.conda/envs/bilby/bin/bilby_pipe_analysis outdir_13.02.fast_tutorial/13.02.fast_tutorial_config_complete.ini --outdir outdir_13.02.fast_tutorial --detectors H1 --detectors L1 --label 13-02-fast_tutorial_data0_0_analysis_H1L1_par1 --data-dump-file outdir_13.02.fast_tutorial/data/13-02-fast_tutorial_data0_0_generation_data_dump.pickle --sampler dynesty"
    /cluster/users/venkatek/.conda/envs/bilby/bin/bilby_pipe_analysis outdir_13.02.fast_tutorial/13.02.fast_tutorial_config_complete.ini --outdir outdir_13.02.fast_tutorial --detectors H1 --detectors L1 --label 13-02-fast_tutorial_data0_0_analysis_H1L1_par1 --data-dump-file outdir_13.02.fast_tutorial/data/13-02-fast_tutorial_data0_0_generation_data_dump.pickle --sampler dynesty
fi

# 13-02-fast_tutorial_data0_0_analysis_H1L1_merge
# PARENTS 13-02-fast_tutorial_data0_0_analysis_H1L1_par0 13-02-fast_tutorial_data0_0_analysis_H1L1_par1
# CHILDREN 13-02-fast_tutorial_data0_0_analysis_H1L1_merge_final_result
if [[ "13-02-fast_tutorial_data0_0_analysis_H1L1_merge" == *"$1"* ]]; then
    echo "Running: /cluster/users/venkatek/.conda/envs/bilby/bin/bilby_result --result outdir_13.02.fast_tutorial/result/13-02-fast_tutorial_data0_0_analysis_H1L1_par0_result.hdf5 outdir_13.02.fast_tutorial/result/13-02-fast_tutorial_data0_0_analysis_H1L1_par1_result.hdf5 --outdir outdir_13.02.fast_tutorial/result --label 13-02-fast_tutorial_data0_0_analysis_H1L1_merge --extension hdf5 --merge"
    /cluster/users/venkatek/.conda/envs/bilby/bin/bilby_result --result outdir_13.02.fast_tutorial/result/13-02-fast_tutorial_data0_0_analysis_H1L1_par0_result.hdf5 outdir_13.02.fast_tutorial/result/13-02-fast_tutorial_data0_0_analysis_H1L1_par1_result.hdf5 --outdir outdir_13.02.fast_tutorial/result --label 13-02-fast_tutorial_data0_0_analysis_H1L1_merge --extension hdf5 --merge
fi

# 13-02-fast_tutorial_data0_0_analysis_H1L1_merge_final_result
# PARENTS 13-02-fast_tutorial_data0_0_analysis_H1L1_merge
# CHILDREN 
if [[ "13-02-fast_tutorial_data0_0_analysis_H1L1_merge_final_result" == *"$1"* ]]; then
    echo "Running: /cluster/users/venkatek/.conda/envs/bilby/bin/bilby_result --result outdir_13.02.fast_tutorial/result/13-02-fast_tutorial_data0_0_analysis_H1L1_merge_result.hdf5 --outdir outdir_13.02.fast_tutorial/final_result --extension hdf5 --max-samples 20000 --lightweight --save"
    /cluster/users/venkatek/.conda/envs/bilby/bin/bilby_result --result outdir_13.02.fast_tutorial/result/13-02-fast_tutorial_data0_0_analysis_H1L1_merge_result.hdf5 --outdir outdir_13.02.fast_tutorial/final_result --extension hdf5 --max-samples 20000 --lightweight --save
fi

