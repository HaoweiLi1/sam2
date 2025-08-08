#!/bin/bash

# Create a log directory for outputs
LOG_DIR="./mask_generation_logs"
mkdir -p $LOG_DIR

# Log file path with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/sam2_mask_generation_${TIMESTAMP}.log"

# Add header to log file
echo "Starting SAM2 mask generation job at $(date)" > $MAIN_LOG
echo "=========================================" >> $MAIN_LOG

# Base directory
BASE_INPUT_DIR="/media/hdd2/users/haowei/Dataset/Replica"
OUTPUT_BASE_DIR="/media/hdd2/users/haowei/Dataset/sam2"
mkdir -p $OUTPUT_BASE_DIR

# Scene directories
SCENES=("office3" "room1")

# SAM2 paths (ensure these are correct)
SAM2_CHECKPOINT="checkpoints/sam2.1_hiera_large.pt"
MODEL_CONFIG="configs/sam2.1/sam2.1_hiera_l.yaml"

# Available GPUs
GPUS=(1 3)
NUM_GPUS=${#GPUS[@]}

# Arrays to store commands for each GPU
GPU1_COMMANDS=()
GPU3_COMMANDS=()

# Current GPU index for round-robin assignment
CURRENT_GPU=0

# Generate commands for all scenes and assign to GPUs
for SCENE in "${SCENES[@]}"; do
    # Input directory path
    INPUT_DIR="${BASE_INPUT_DIR}/${SCENE}/frames"
    
    # Output directory path
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${SCENE}/mask"
    mkdir -p $OUTPUT_DIR
    
    # Skip if input directory doesn't exist
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Skipping due to missing input directory: $INPUT_DIR" >> $MAIN_LOG
        continue
    fi
    
    # Assign to current GPU (round-robin)
    GPU_ID=${GPUS[$CURRENT_GPU]}
    
    # Log file for this specific task
    TASK_LOG="${LOG_DIR}/${SCENE}_gpu${GPU_ID}_${TIMESTAMP}.log"
    
    # Create command with CUDA_VISIBLE_DEVICES to target specific GPU
    CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python frame_to_mask.py --input-dir ${INPUT_DIR} --output-dir ${OUTPUT_DIR} --checkpoint ${SAM2_CHECKPOINT} --config ${MODEL_CONFIG} > ${TASK_LOG} 2>&1"
    
    # Add to GPU-specific commands array
    if [ "$GPU_ID" -eq 1 ]; then
        GPU0_COMMANDS+=("$CMD")
    elif [ "$GPU_ID" -eq 3 ]; then
        GPU1_COMMANDS+=("$CMD")
    fi
    
    # Log the assignment
    echo "Assigned task: ${SCENE} to GPU ${GPU_ID}" >> $MAIN_LOG
    
    # Move to next GPU (round-robin)
    CURRENT_GPU=$(( (CURRENT_GPU + 1) % NUM_GPUS ))
done

# Function to run tasks on a specific GPU
run_gpu_tasks() {
    local GPU_ID=$1
    local COMMANDS=()
    
    # Select the correct command array based on GPU ID
    if [ "$GPU_ID" -eq 1 ]; then
        COMMANDS=("${GPU0_COMMANDS[@]}")
    elif [ "$GPU_ID" -eq 3 ]; then
        COMMANDS=("${GPU1_COMMANDS[@]}")
    fi
    
    # Process one task at a time on each GPU
    local MAX_JOBS=1  # Each GPU handles 1 scene at a time
    
    echo "Starting execution on GPU ${GPU_ID} at $(date)" >> $MAIN_LOG
    echo "Number of tasks for GPU ${GPU_ID}: ${#COMMANDS[@]}" >> $MAIN_LOG
    
    # Process each command for this GPU sequentially
    for CMD in "${COMMANDS[@]}"; do
        echo "Starting new task on GPU ${GPU_ID}" >> $MAIN_LOG
        eval "$CMD"
        echo "Task completed on GPU ${GPU_ID}" >> $MAIN_LOG
    done
    
    echo "All tasks on GPU ${GPU_ID} completed at $(date)" >> $MAIN_LOG
}

# Start processing on each GPU in parallel
for GPU_ID in "${GPUS[@]}"; do
    (run_gpu_tasks $GPU_ID >> $MAIN_LOG 2>&1) &
    MAIN_PID=$!
    echo "Started processing on GPU ${GPU_ID} with PID $MAIN_PID" >> $MAIN_LOG
    echo "Started processing on GPU ${GPU_ID} with PID $MAIN_PID"
    # Brief sleep to stagger startup
    sleep 2
done

echo "All GPU processing started. Log file: $MAIN_LOG"
echo "You can now disconnect from the server. The processes will continue running."
echo "To check status later, use: tail -f $MAIN_LOG"