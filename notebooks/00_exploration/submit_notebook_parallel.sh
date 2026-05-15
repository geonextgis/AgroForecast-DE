#!/bin/bash

# --- List of crops ---
crops=("winter_wheat" "winter_barley" "winter_rye" "winter_rapeseed" "silage_maize")

# --- Submit a separate job for each crop ---
for crop in "${crops[@]}"; do
    echo "Submitting SLURM job for $crop..."
    sbatch --export=CROP=$crop submit_notebook.sh
done
