#!/bin/bash

# Check if the 'env' directory exists
if [ ! -d "env" ]; then
  echo "Creating virtual environment..."
  python3 -m venv env
	source env/bin/activate
	pip install -r requirements.txt
else
	source env/bin/activate
fi

echo "Virtual environment activated."

