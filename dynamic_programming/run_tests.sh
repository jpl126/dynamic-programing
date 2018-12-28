#!/usr/bin/env bash

CURRENT_DIR=$(pwd)
cd $(dirname "$0")

python -m pytest .

cd "$CURRENT_DIR"