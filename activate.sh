#!/bin/bash

# Check if the 'env' directory exists
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt
else
	source venv/bin/activate
fi

echo "Virtual environment activated."

