#!/usr/bin/env bash
set -e
mkdir -p figs
python run_base.py
python run_mc.py
python run_real.py
python run_washout.py
python figs.py
