#!/bin/bash
#SBATCH --job-name=baseline_${CROP}
#SBATCH --output=baseline_${CROP}_%j.log
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# --- Initialize conda in non-interactive shell ---
source ~/.bashrc
conda activate geo

echo "🚀 Running baseline for $CROP..."

# --- Run the notebook ---
OUTPUT_NOTEBOOK="00_yield_data_exploration_${CROP}.ipynb"
papermill 00_yield_data_exploration.ipynb "$OUTPUT_NOTEBOOK" -p CROP "$CROP"

echo "✅ Completed baseline for $CROP"

# --- Delete generated notebook ---
rm -f "$OUTPUT_NOTEBOOK"
echo "🗑️ Deleted $OUTPUT_NOTEBOOK"
