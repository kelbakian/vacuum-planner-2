#!/bin/bash
ALGO=$1
BATTERY=${2:-False}
HEURISTIC=${3:-None}
python a2.py $@ <&0
