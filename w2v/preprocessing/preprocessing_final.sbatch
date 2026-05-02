#!/bin/bash -l

#SBATCH --job-name=preprocessing_de
#SBATCH --output=/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/w2v/preprocessing/logs/preprocessing_final_de.out
#SBATCH --error=/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/w2v/preprocessing/logs/preprocessing_final_de.err
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --mem=120G
#SBATCH --partition=ssd
#SBATCH --account=ssd
#SBATCH --time=04:00:00

# Create logs directory if it doesn't exist
mkdir -p /project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/w2v/preprocessing/logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Number of tasks (parallel processes): $SLURM_NTASKS"

# Load Python environment
module load python/anaconda-2022.05
conda activate ideological_resonance_thesis

# Change to working directory
cd /project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/w2v/preprocessing

# Run the preprocessing script. _en or _de depending on the language.
python preprocessing_final_de.py

# Print completion info
echo "Job completed at: $(date)"