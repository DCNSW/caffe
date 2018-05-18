#! /usr/bin/env sh
set -e

./build/tools/caffe train -solver \
  ../examples/Resnet/model/ResNet_solver.prototxt -weights \
  ../examples/Resnet/model/snapshot/ResNet-50-model.caffemodel \
  2>&1 | tee res.log
