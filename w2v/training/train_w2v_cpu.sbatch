#!/bin/bash -l

#SBATCH --job-name=w2v_training
#SBATCH --output=/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/w2v/training/logs/train_w2v_cpu_2.out
#SBATCH --error=/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/w2v/training/logs/train_w2v_cpu_2.err
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=80G
#SBATCH --partition=ssd
#SBATCH --account=ssd
#SBATCH --time=04:00:00

# ============================================================================
# CONFIGURATION: Edit these variables before running
# ============================================================================

# Input/Output paths
INPUT_FILE="/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/w2v/data/clean_final_en.csv"
OUTPUT_MODEL="/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/w2v/models/2_w2v_min10"

# Model hyperparameters
VECTOR_SIZE=300        # Embedding dimension (common: 100, 200, 300)
WINDOW=8               # Context window size (common: 5-10)
MIN_COUNT=10            # Minimum word frequency threshold
EPOCHS=10               # Number of training epochs
NEGATIVE=5             # Number of negative samples for skip-gram

# Technical parameters
WORKERS=16             # Should match --ntasks above
SEED=42                # Random seed for reproducibility

# Data parameters
TEXT_COLUMN="cleaned_text"    # Column name with cleaned text
MIN_SENTENCE_LENGTH=2         # Minimum tokens per sentence

# Output options (set to "true" to enable)
GENERATE_PLOTS="true"         # Generate training overview plots
VERBOSE="true"                # Enable verbose logging

# ============================================================================

# Create logs directory if it doesn't exist
mkdir -p /project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/w2v/training/logs

# Print job information
echo "========================================================"
echo "Word2Vec Training Job"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Number of tasks (workers): $SLURM_NTASKS"
echo "Memory allocated: 80GB"
echo ""
echo "Configuration:"
echo "  Input file: $INPUT_FILE"
echo "  Output model: $OUTPUT_MODEL"
echo "  Vector size: $VECTOR_SIZE"
echo "  Window: $WINDOW"
echo "  Min count: $MIN_COUNT"
echo "  Epochs: $EPOCHS"
echo "  Workers: $WORKERS"
echo "========================================================"
echo ""

# Load Python environment
module load python/anaconda-2022.05
conda activate ideological_resonance_thesis

# Change to working directory
cd /project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/w2v/training

# Build the command with optional flags
CMD="python train_w2v_cpu.py \
    --input $INPUT_FILE \
    --output $OUTPUT_MODEL \
    --vector-size $VECTOR_SIZE \
    --window $WINDOW \
    --min-count $MIN_COUNT \
    --epochs $EPOCHS \
    --negative $NEGATIVE \
    --workers $WORKERS \
    --seed $SEED \
    --text-column $TEXT_COLUMN \
    --min-sentence-length $MIN_SENTENCE_LENGTH"

# Add optional flags
if [ "$GENERATE_PLOTS" = "true" ]; then
    CMD="$CMD --plot"
fi

if [ "$VERBOSE" = "true" ]; then
    CMD="$CMD --verbose"
fi

# Run the training script
echo "Running command:"
echo "$CMD"
echo ""
eval $CMD

# Print completion info
echo ""
echo "========================================================"
echo "Job completed at: $(date)"
echo "========================================================"