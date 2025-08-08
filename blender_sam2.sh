#!/bin/bash

SCENES=("office0" "office1" "office2" "office3" "office4" "room0" "room1" "room2")

LOGS_DIR="./blender_logs"
mkdir -p $LOGS_DIR

# Maximum number of concurrent processes
MAX_PARALLEL=4

# Array to store process IDs
pids=()

process_scene() {
    scene=$1
    echo "Starting processing for scene: $scene"
    python blender_sam2.py --scenes $scene > "$LOGS_DIR/${scene}_log.txt" 2>&1
    echo "Completed processing for scene: $scene"
}

# Process scenes in batches of MAX_PARALLEL
echo "Starting processing of scenes with max $MAX_PARALLEL at a time..."
for scene in "${SCENES[@]}"; do
    # If we already have MAX_PARALLEL processes running, wait for one to finish
    if [ ${#pids[@]} -ge $MAX_PARALLEL ]; then
        # Wait for any process to finish
        wait -n
        
        # Remove completed processes from the pids array
        new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 $pid 2>/dev/null; then
                new_pids+=($pid)
            fi
        done
        pids=("${new_pids[@]}")
    fi
    
    # Start a new process and store its PID
    process_scene "$scene" &
    pids+=($!)
    echo "Started process for $scene (PID: $!), current processes: ${#pids[@]}/$MAX_PARALLEL"
done

# Wait for all remaining processes to complete
wait
echo "All processing complete!"