#!/bin/bash

case $1 in
  venv)
    source .venv/Scripts/activate
    ;;
  test)
    echo "Starting testing..."
    python main.py --test
    ;;
  *)
    echo "Usage: $0 {venv|test}"
    exit 1
    ;;
esac