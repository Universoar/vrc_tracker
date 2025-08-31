#!/bin/bash

case $1 in
  venv)
    source .venv/Scripts/activate
    ;;
  gen)
    pipreqs . --force --ignore .venv
    ;;
  install)
    pip install -r requirements.txt
    ;;
  test)
    echo "Starting testing..."
    python main.py --test
    ;;
  *)
    echo ""
    echo "venv      -      Create and activate a virtual environment"
    echo "gen       -      Generate requirements.txt"
    echo "install   -      Install dependencies"
    echo "test      -      Run tests"
    exit 1
    ;;
esac