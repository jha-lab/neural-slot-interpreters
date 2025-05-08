
echo "Downloading CLEVRTex dataset..."

# Download all parts to datasets folder
wget https://thor.robots.ox.ac.uk/datasets/clevrtex/clevrtexv2_full.tar

echo "Extracting CLEVRTex dataset..."

# Extract all parts
tar -xzf clevrtexv2_full.tar

# Optional: Remove archives after extraction
rm clevrtexv2_full.tar

echo "Building CLEVRTex dataset..."

python build_clevrtex_dataset.py

echo "CLEVRTex dataset built successfully."






