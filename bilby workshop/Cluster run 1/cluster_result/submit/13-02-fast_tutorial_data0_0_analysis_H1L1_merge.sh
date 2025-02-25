#!/bin/bash

source bilby

python /cluster/users/venkatek/.conda/envs/bilby/bin/bilby_result --result outdir_13.02.fast_tutorial/result/13-02-fast_tutorial_data0_0_analysis_H1L1_par0_result.hdf5 outdir_13.02.fast_tutorial/result/13-02-fast_tutorial_data0_0_analysis_H1L1_par1_result.hdf5 --outdir outdir_13.02.fast_tutorial/result --label 13-02-fast_tutorial_data0_0_analysis_H1L1_merge --extension hdf5 --merge

