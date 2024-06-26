#!/bin/bash
#SBATCH -J paac
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=2
#SBATCH -t 5:00:00
#SBATCH --gres=gpu:H100:2 
#SBATCH --mem-per-gpu=32G
#SBATCH -o ./slurm_outs/slurm_%j.out

cd ~./PAAC
module purge
module load anaconda3/2022.05.0.1
conda deactivate
conda activate PAAC
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "Number of CPUs: $SLURM_CPUS_ON_NODE"

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo 'MASTER_PORT=$MASTER_PORT'
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export CUDA_VISIBLE_DEVICES=0,1
python main.py --env-name BreakoutDeterministic-v4 
# this is adjusted in python: --num-workers $num_workers --num-envs $num_envs