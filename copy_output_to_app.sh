#!/bin/bash
# Script to copy output directory to iOS simulator's document directory

OUTPUT_DIR="/Users/jmoran0/swanhacks2025/output"
SIMULATOR_DATA=$(xcrun simctl get_app_container booted com.notes.notes data 2>/dev/null || echo "")

if [ -z "$SIMULATOR_DATA" ]; then
    echo "Could not find simulator data directory. Make sure the app is running in the simulator."
    exit 1
fi

DOCUMENTS_DIR="$SIMULATOR_DATA/Documents/output"

echo "Copying output directory to simulator..."
echo "Source: $OUTPUT_DIR"
echo "Destination: $DOCUMENTS_DIR"

mkdir -p "$DOCUMENTS_DIR"
cp -r "$OUTPUT_DIR"/* "$DOCUMENTS_DIR/" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "Successfully copied output directory!"
else
    echo "Failed to copy. Trying alternative method..."
    # Alternative: find the simulator directory
    SIMULATORS=$(xcrun simctl list devices | grep Booted | head -1)
    echo "Please manually copy files or use the app's document directory"
fi
