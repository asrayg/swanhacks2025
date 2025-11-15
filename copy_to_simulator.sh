#!/bin/bash
# Script to copy output directory to iOS simulator's document directory

OUTPUT_SOURCE="/Users/jmoran0/swanhacks2025/output"

# Get the running simulator's data directory
SIMULATOR_ID=$(xcrun simctl list devices | grep Booted | head -1 | sed 's/.*(\([^)]*\)).*/\1/')

if [ -z "$SIMULATOR_ID" ]; then
    echo "No booted simulator found. Please start the app in the simulator first."
    exit 1
fi

# Find the app's data directory
APP_DATA=$(xcrun simctl get_app_container "$SIMULATOR_ID" com.notes.notes data 2>/dev/null)

if [ -z "$APP_DATA" ]; then
    echo "Could not find app data directory. Make sure the app is installed and running."
    exit 1
fi

DOCUMENTS_DIR="$APP_DATA/Documents/output"

echo "Copying output directory to simulator..."
echo "Source: $OUTPUT_SOURCE"
echo "Destination: $DOCUMENTS_DIR"

mkdir -p "$DOCUMENTS_DIR"
cp -r "$OUTPUT_SOURCE"/* "$DOCUMENTS_DIR/" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "Successfully copied output directory to simulator!"
    echo "Files are now available in the app."
else
    echo "Failed to copy files."
    exit 1
fi
