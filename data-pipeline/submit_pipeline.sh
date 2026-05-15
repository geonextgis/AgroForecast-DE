#!/bin/bash
#SBATCH --job-name=data_pipeline
#SBATCH --output=data_pipeline_%j.log
#SBATCH --time=48:00:00
#SBATCH --partition=highmem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80

# =============================================================================
# AgroForecast-DE — data-pipeline submit script
# =============================================================================
# Runs the data-pipeline steps sequentially in a single SLURM job.
# Toggle a step off by setting RUN_XX=0 (env var or edit below).
#
# Usage:
#   sbatch data-pipeline/submit_pipeline.sh
#
#   # Skip Step 7 and override years:
#   sbatch --export=ALL,RUN_06=0,YEAR_START=2001,YEAR_END=2024 \
#       data-pipeline/submit_pipeline.sh
# =============================================================================

# --- Step toggles ------------------------------------------------------------
RUN_01="${RUN_01:-0}"   # HYRAS NetCDF → district daily
RUN_02="${RUN_02:-0}"   # HOSTRADA hourly → district daily
RUN_03="${RUN_03:-1}"   # Derived vars (VPD, CWB)
RUN_04="${RUN_04:-1}"   # Assemble per-district Parquet
RUN_05="${RUN_05:-1}"   # GCFS2.2 seasonal forecast
RUN_06="${RUN_06:-0}"   # Static covariates (one-time; off by default)

# --- Shared parameters -------------------------------------------------------
YEAR_START="${YEAR_START:-2000}"
YEAR_END="${YEAR_END:-2026}"
SFC_DATE="${SFC_DATE:-20260401}"

DATA_ROOT="${DATA_ROOT:-/beegfs/halder/GITHUB/RESEARCH/AgroForecast-DE/data}"

DISTRICT_SHP="${DISTRICT_SHP:-${DATA_ROOT}/raw/districts/districts.gpkg}"

OVERWRITE="${OVERWRITE:-0}"

CLIMATE_INTERIM="${DATA_ROOT}/interim/climate"

OVERWRITE_FLAG=""
[[ "$OVERWRITE" == "1" ]] && OVERWRITE_FLAG="--overwrite"

# --- Environment -------------------------------------------------------------
source ~/.bashrc
conda activate geo

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "================================================================="
echo "AgroForecast-DE data pipeline"
echo "  Years      : ${YEAR_START}-${YEAR_END}"
echo "  GCFS2.2    : ${SFC_DATE}"
echo "  Data root  : ${DATA_ROOT}"
echo "  Districts  : ${DISTRICT_SHP}"
echo "  Overwrite  : ${OVERWRITE}"
echo "  Steps      : 01=${RUN_01} 02=${RUN_02} 03=${RUN_03}"
echo "               04=${RUN_04} 05=${RUN_05} 06=${RUN_06}"
echo "================================================================="

run_step () {
    local label="$1"; shift

    echo ""
    echo ">>> [${label}] $*"
    echo ">>> started: $(date -Is)"

    "$@"

    echo ">>> finished: $(date -Is)"
}

# =============================================================================
# Step 1 — HYRAS NetCDF → district daily
# =============================================================================
if [[ "$RUN_01" == "1" ]]; then
    run_step "01_process_hyras" \
        python 01_process_hyras.py \
            --variables tas tasmax tasmin hurs pr evpot \
            --years "$YEAR_START" "$YEAR_END" \
            --district_shapefile "$DISTRICT_SHP" \
            --output_dir "${CLIMATE_INTERIM}/hyras" \
            $OVERWRITE_FLAG
else
    echo "--- skipping 01_process_hyras"
fi

# =============================================================================
# Step 2 — HOSTRADA hourly → district daily
# =============================================================================
if [[ "$RUN_02" == "1" ]]; then
    run_step "02_process_hostrada" \
        python 02_process_hostrada.py \
            --variables rsds sfcWind \
            --years "$YEAR_START" "$YEAR_END" \
            --district_shapefile "$DISTRICT_SHP" \
            --output_dir "${CLIMATE_INTERIM}/hostrada" \
            $OVERWRITE_FLAG
else
    echo "--- skipping 02_process_hostrada"
fi

# =============================================================================
# Step 3 — Derived variables (VPD, CWB)
# =============================================================================
if [[ "$RUN_03" == "1" ]]; then
    run_step "03_compute_derived_vars" \
        python 03_compute_derived_vars.py \
            --years "$YEAR_START" "$YEAR_END" \
            --interim_dir "${CLIMATE_INTERIM}" \
            --output_dir "${CLIMATE_INTERIM}/derived" \
            $OVERWRITE_FLAG
else
    echo "--- skipping 03_compute_derived_vars"
fi

# =============================================================================
# Step 4 — Assemble per-district Parquet
# =============================================================================
if [[ "$RUN_04" == "1" ]]; then
    run_step "04_assemble_district_files" \
        python 04_assemble_district_files.py \
            --years "$YEAR_START" "$YEAR_END" \
            --interim_dir "${CLIMATE_INTERIM}" \
            --output_dir "${CLIMATE_INTERIM}/districts" \
            $OVERWRITE_FLAG
else
    echo "--- skipping 04_assemble_district_files"
fi

# =============================================================================
# Step 5 — GCFS2.2 seasonal forecast
# =============================================================================
if [[ "$RUN_05" == "1" ]]; then
    run_step "05_ingest_gcfs22_forecast" \
        python 05_ingest_gcfs22_forecast.py \
            --sfc_date "$SFC_DATE" \
            --district_shapefile "$DISTRICT_SHP" \
            --output_dir "${DATA_ROOT}/interim/gcfs22" \
            $OVERWRITE_FLAG
else
    echo "--- skipping 05_ingest_gcfs22_forecast"
fi

# =============================================================================
# Step 6 — Static covariates
# =============================================================================
if [[ "$RUN_06" == "1" ]]; then
    run_step "06_ingest_static" \
        python 06_ingest_static.py \
            --district_shapefile "$DISTRICT_SHP" \
            --output_dir "${DATA_ROOT}/interim/static" \
            $OVERWRITE_FLAG
else
    echo "--- skipping 06_ingest_static"
fi

echo ""
echo "================================================================="
echo "Data pipeline complete."
echo "================================================================="