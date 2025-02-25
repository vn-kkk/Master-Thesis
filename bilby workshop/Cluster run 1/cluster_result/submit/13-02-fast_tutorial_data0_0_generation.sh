#!/bin/bash

source bilby

python /cluster/users/venkatek/.conda/envs/bilby/bin/bilby_pipe_generation outdir_13.02.fast_tutorial/13.02.fast_tutorial_config_complete.ini --label 13-02-fast_tutorial_data0_0_generation --idx 0 --trigger-time 0 --injection-file outdir_13.02.fast_tutorial/data/13.02.fast_tutorial_injection_file.dat --outdir outdir_13.02.fast_tutorial

