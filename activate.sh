#!/bin/bash

# Check if the 'env' directory exists.

# If it does not exist, create a virtual environment .
# Then activate it and install the requirements.
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt

# If it exists, activate the virtual environment.
else
	source venv/bin/activate
fi

echo "Virtual environment activated."

