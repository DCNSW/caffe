#!/usr/bin/env sh
set -e

TOOLS=$CAFFE_ROOT/build/tools/caffe 
$TOOLS train \
    --solver=$CIFAR_ROOT/model/cifar10_full/cifar10_full_solver.prototxt $@

## reduce learning rate by factor of 10
#$TOOLS/caffe train \
#    --solver=examples/cifar10/cifar10_full_solver_lr1.prototxt \
#    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate $@
#
## reduce learning rate by factor of 10
#$TOOLS/caffe train \
#    --solver=examples/cifar10/cifar10_full_solver_lr2.prototxt \
#    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate $@
