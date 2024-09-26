#!/bin/sh
#SBATCH -J  wanda       # Job name
#SBATCH -o ./out/llama-2-7b/%j.out      # Name of stdout output file (%j expands to %jobId)
#SBATCH -t 3-00:00:00         # Run time (hh:mm:ss)

#### Select  GPU
#SBATCH -p 3090               # partiton
#SBATCH --nodes=1           # number of nodes
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4     # number of cpus

# >>> Number of GPUs <<< #df
#SBATCH --gres=gpu:1

cd  $SLURM_SUBMIT_DIR

srun -I /bin/hostname
srun -I /bin/pwd
srun -I /bin/date

## Load modules
module purge
module load cuda/12.1
module load cudnn/cuda-12.1/8.9.7

##  Python  Virtual Env ##

echo "Start"
echo "condaPATH"
export HF_DATASETS_TRUST_REMOTE_CODE=1

echo "source $HOME/anaconda3/etc/profile.d/conda.sh"
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh    #anaconda path

echo "conda activate wanda"

conda activate wanda    #conda environment to use \

model="decapoda-research/llama-7b-hf"
method="wanda" # select method from [magnitude,wanda,sparsegpt]
sparsity_ratio=0.5 # select sparsity ratio
cuda_device=0
sparsity_type="unstructured" # [unstructured,2:4,4:8]

python main.py \
    --model $model \
    --prune_method $method \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $sparsity_type \
    --save out/$model/$sparsity_type/$method/


date

echo " conda deactivate wanda"

conda deactivate #deactivate environment

squeue --job $SLURM_JOBID

echo  "##### END #####"