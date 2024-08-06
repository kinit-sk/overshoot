#!/bin/bash
# set -xe

JOB_NAME=${1:?"Missing experiment name"}
OVERSHOOT_FACTORS=(1.0 2.0 4.0 6.0 10.0 16.0 24.0)

DST="lightning_logs/${JOB_NAME}"
if [ -d "${DST}" ]; then
    echo "Removing previous experiments results!"
    rm -rf "${DST}"
fi
mkdir -p "${DST}"
mkdir -p "slurm_logs"

cp overshoot.py "lightning_logs/${JOB_NAME}/"
cp datasets.py "lightning_logs/${JOB_NAME}/"
cp overshoot-job.sh "lightning_logs/${JOB_NAME}/"
cp overshoot-batch.sh "lightning_logs/${JOB_NAME}/"

for FACTOR in "${OVERSHOOT_FACTORS[@]}"; do
    sbatch --output="slurm_logs/${JOB_NAME}_#_factor:_${FACTOR}.job" -J "${JOB_NAME}"  --export=ALL,JOB_NAME=${JOB_NAME},OVERSHOOT_FACTOR=${FACTOR} devana-job.sh 
done