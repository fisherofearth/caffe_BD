#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/BSDS500/BSDS500_train_solver.prototxt

