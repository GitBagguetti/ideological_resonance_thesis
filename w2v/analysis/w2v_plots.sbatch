#!/bin/bash -l

#SBATCH --job-name=w2v_plots
#SBATCH --output=/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/w2v/analysis/logs/w2v_plots.out
#SBATCH --error=/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/w2v/analysis/logs/w2v_plots.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --partition=ssd
#SBATCH --account=ssd
#SBATCH --time=02:00:00

# Load Python environment
module load python
conda activate word2vec

SCRIPT="/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/w2v/analysis/modules/semanalysis_plots.py"

echo "Running EN–GE semantic / PCA plots pipeline at: $(date)"

# ['aligned_correlation', 'correlation', 'correlation_abs', 'inverse_squared_distance', 
# 'kendall_p', 'kendall_tau', 'kendall_tau_abs', 'mean_absolute_error', 'spearman_p', 
# 'spearman_rho', 'spearman_rho_abs', 'sum_squared_distance', 'top_k', 'top_k_overlap']
python "$SCRIPT" --metric spearman_rho --flip_axis pca_trump_anchor --crossaxis reveal_holy

echo ""
echo "Job completed at: $(date)"
