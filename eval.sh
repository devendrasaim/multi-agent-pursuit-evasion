#!/bin/bash

# Define the root and sub directories
ROOT_DIR="$(pwd)"
SUB_DIR="$ROOT_DIR/submissions"

# Loop through each folder inside 'sub'
for folder in "$SUB_DIR"/*/; do
    if [ -d "$folder" ]; then
        # Extract the subdirectory name
        SUBDIR_NAME=$(basename "$folder")

        echo "Processing folder: $SUBDIR_NAME"

        # Copy the main.py from the subfolder to root
        cp "$folder/planner.py" "$ROOT_DIR/planner.py"

        # Execute the main.py in root with the folder name as an argument
        python3 "$ROOT_DIR/main.py" --name "$SUBDIR_NAME"

        echo "Finished executing main.py from $SUBDIR_NAME"
    fi
done
