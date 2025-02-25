#!/bin/bash

source bilby

python /cluster/users/venkatek/.conda/envs/bilby/bin/bilby_pipe_analysis outdir_13.02.fast_tutorial/13.02.fast_tutorial_config_complete.ini --outdir outdir_13.02.fast_tutorial --detectors H1 --detectors L1 --label 13-02-fast_tutorial_data0_0_analysis_H1L1_par1 --data-dump-file outdir_13.02.fast_tutorial/data/13-02-fast_tutorial_data0_0_generation_data_dump.pickle --sampler dynesty

