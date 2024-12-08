#!/bin/bash

# Check if the user provided a command to run
if [ $# -lt 1 ]; then
    echo "Usage: $0 '<program>' [program arguments...]"
    exit 1
fi

# Set default Nsight Systems options
NSYS_OPTIONS="--stats=true --trace=cuda,osrt,nvtx --output=profile_xavier_nx"

# Run nsys with the provided program command
echo "Running Nsight Systems with the following command:"
echo "nsys profile $NSYS_OPTIONS $@"
nsys profile $NSYS_OPTIONS "$@" 
