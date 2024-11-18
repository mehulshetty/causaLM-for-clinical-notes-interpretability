#!/bin/bash

# Test submission for directing output to the 'output' directory

# Base parameters
ACCOUNT="swabhas_1457"
PROJECT_DIR="$HOME/causaLM-for-clinical-notes-interpretability"
OUTPUT_DIR="$HOME/causaLM-output"
JOB_SCRIPT="$PROJECT_DIR/run_causaLM.job"

# Number of test jobs
TEST_JOBS=1

JOB_ID=1

# Ensure the output directory exists
mkdir -p "${OUTPUT_DIR}"

# Function to submit a test job
submit_test_job() {
    local job_name=$1
    local mem=$2
    local gpus=$3
    local time_limit=$4
    local index=$5
    local job_id=$6
    local output_file=$7

    sbatch \
        --job-name="${job_name}_${index}" \
        --mem=${mem} \
        --gres=gpu:${gpus} \
        --time=${time_limit} \
        --account=${ACCOUNT} \
        --output="${output_file}" \
        --export=JOB_ID=${job_id} \
        ${JOB_SCRIPT}
}

# Submit a single test job
JOB_NAME="causaLM_test"
MEM="16G"
GPUS="1"
TIME_LIMIT="1:00:00"
INDEX=1
OUTPUT_FILE="${OUTPUT_DIR}/output_test_${INDEX}.log"
submit_test_job "${JOB_NAME}" "${MEM}" "${GPUS}" "${TIME_LIMIT}" "${INDEX}" "${JOB_ID}" "${OUTPUT_FILE}"

echo "Test job has been submitted."
