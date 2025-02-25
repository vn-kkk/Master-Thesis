#!/bin/bash
#SBATCH --mem=1G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=outdir_13.02.fast_tutorial/submit/13.02.fast_tutorial_master_slurm.out
#SBATCH --error=outdir_13.02.fast_tutorial/submit/13.02.fast_tutorial_master_slurm.err
#SBATCH --job-name=13.02.fast_tutorial_master
#SBATCH --partition=all

jid0=($(sbatch --mem=8G --nodes=1 --ntasks-per-node=1 --time=1:00:00 --output=outdir_13.02.fast_tutorial/log_data_generation/13-02-fast_tutorial_data0_0_generation.out --error=outdir_13.02.fast_tutorial/log_data_generation/13-02-fast_tutorial_data0_0_generation.err --job-name=13-02-fast_tutorial_data0_0_generation --partition=all outdir_13.02.fast_tutorial/submit/13-02-fast_tutorial_data0_0_generation.sh))

echo "jid0 ${jid0[-1]}" >> outdir_13.02.fast_tutorial/submit/slurm_ids

jid1=($(sbatch --mem=8G --nodes=1 --ntasks-per-node=4 --time=7-00:00:00 --output=outdir_13.02.fast_tutorial/log_data_analysis/13-02-fast_tutorial_data0_0_analysis_H1L1_par0.out --error=outdir_13.02.fast_tutorial/log_data_analysis/13-02-fast_tutorial_data0_0_analysis_H1L1_par0.err --job-name=13-02-fast_tutorial_data0_0_analysis_H1L1_par0 --partition=all --dependency=afterok:${jid0[-1]} outdir_13.02.fast_tutorial/submit/13-02-fast_tutorial_data0_0_analysis_H1L1_par0.sh))

echo "jid1 ${jid1[-1]}" >> outdir_13.02.fast_tutorial/submit/slurm_ids

jid2=($(sbatch --mem=8G --nodes=1 --ntasks-per-node=4 --time=7-00:00:00 --output=outdir_13.02.fast_tutorial/log_data_analysis/13-02-fast_tutorial_data0_0_analysis_H1L1_par1.out --error=outdir_13.02.fast_tutorial/log_data_analysis/13-02-fast_tutorial_data0_0_analysis_H1L1_par1.err --job-name=13-02-fast_tutorial_data0_0_analysis_H1L1_par1 --partition=all --dependency=afterok:${jid0[-1]} outdir_13.02.fast_tutorial/submit/13-02-fast_tutorial_data0_0_analysis_H1L1_par1.sh))

echo "jid2 ${jid2[-1]}" >> outdir_13.02.fast_tutorial/submit/slurm_ids

jid3=($(sbatch --mem=16G --nodes=1 --ntasks-per-node=1 --time=1:00:00 --output=outdir_13.02.fast_tutorial/log_data_analysis/13-02-fast_tutorial_data0_0_analysis_H1L1_merge.out --error=outdir_13.02.fast_tutorial/log_data_analysis/13-02-fast_tutorial_data0_0_analysis_H1L1_merge.err --job-name=13-02-fast_tutorial_data0_0_analysis_H1L1_merge --partition=all --dependency=afterok:${jid1[-1]}:${jid2[-1]} outdir_13.02.fast_tutorial/submit/13-02-fast_tutorial_data0_0_analysis_H1L1_merge.sh))

echo "jid3 ${jid3[-1]}" >> outdir_13.02.fast_tutorial/submit/slurm_ids

jid4=($(sbatch --mem=4G --nodes=1 --ntasks-per-node=1 --time=1:00:00 --output=outdir_13.02.fast_tutorial/log_data_analysis/13-02-fast_tutorial_data0_0_analysis_H1L1_merge_final_result.out --error=outdir_13.02.fast_tutorial/log_data_analysis/13-02-fast_tutorial_data0_0_analysis_H1L1_merge_final_result.err --job-name=13-02-fast_tutorial_data0_0_analysis_H1L1_merge_final_result --partition=all --dependency=afterok:${jid3[-1]} outdir_13.02.fast_tutorial/submit/13-02-fast_tutorial_data0_0_analysis_H1L1_merge_final_result.sh))

echo "jid4 ${jid4[-1]}" >> outdir_13.02.fast_tutorial/submit/slurm_ids
