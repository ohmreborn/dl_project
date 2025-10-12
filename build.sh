#!/bin/bash

if [ -e ./myenv ];then
	source ./myenv/bin/activate
fi
prefix_path=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'share/cmake/Torch'))")
cmake -B build . -DCMAKE_PREFIX_PATH=$prefix_path
cmake --build build
