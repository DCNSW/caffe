#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=../examples/mnist/build/solver.prototxt 2>&1 | tee mylog.log
#./build/tools/caffe train -solver=../examples/mnist/build/solver.prototxt -snapshot ../examples/mnist/build/snapshot/lenet_iter_5000.solverstate
#./build/tools/caffe test -model ../examples/mnist/build/train_test.prototxt -weights ../examples/mnist/build/snapshot/lenet_iter_5000.caffemodel -gpu all -iterations 100
