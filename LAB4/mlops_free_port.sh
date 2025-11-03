#!/bin/bash

# Find first available port in a range
find_free_port() {
    local start_port=$1
    local end_port=$2
    
    for port in $(seq $start_port $end_port); do
        if ! lsof -i :$port &> /dev/null; then
            echo $port
            return 0
        fi
    done
    
    echo "No free ports found between $start_port and $end_port" >&2
    return 1
}

# Example: Find first free port between 5000 and 5100
free_port=$(find_free_port 5000 5100)

if [ $? -eq 0 ]; then
    echo "Found free port: $free_port"
    # Use the port for your application
    echo "Running application on port $free_port..."
    # Your command here, e.g.:
    # mlflow models serve -m $model_uri -p $free_port --no-conda
fi
