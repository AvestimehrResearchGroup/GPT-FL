#!/bin/bash
set -x  # Enable debugging output

# Download the file
wget -O weight.zip 'https://www.dropbox.com/scl/fi/66t6whm87o33imtu1clb6/weight.zip?rlkey=zlpgne44hev8x31bcdd3au8ay&dl=1'

# Check if wget was successful
if [ $? -eq 0 ]; then
    echo "Download successful, attempting to unzip."
    unzip weight.zip
else
    echo "Download failed, not attempting to unzip."
fi