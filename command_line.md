# Check the job completion status
sacct -j  14773369 --format=JobID,JobName,State,ExitCode,Start,End,Elapsed

# Check if the log files exist
ls -la logs/train_df_14773369.*

# View the output log
cat logs/train_df_14773369.out

# View the error log
cat logs/train_df_14773369.err


sbatch train_asvspoof_df.slurm

squeue -j 14773369