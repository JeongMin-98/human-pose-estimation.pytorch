#!/bin/bash

source ~/.bashrc

BASE_DIR="${WIN_HUMAN_POSE_ESTIMATION_DIR}"
echo "Base directory: $BASE_DIR"
DIRS=("data" "models" "output" "log")

LINK_DIR=$(pwd)

echo $LINK_DIR

# Function to create symbolic links
create_symlink() {
  local dir_name=$1
  local target="$BASE_DIR/$dir_name"
  local link="$LINK_DIR/$dir_name"

  if [ -d "$target" ]; then
    if [ ! -L "$link" ]; then
      ln -s "$target" "$link"
      echo "Created symlink: $link -> $target"
    else
      echo "Symlink already exists: $link"
    fi
  else
    echo "Target directory does not exist: $target"
  fi
}

# Loop through each directory and create symlink
for dir in "${DIRS[@]}"; do
  create_symlink "$dir"
done
