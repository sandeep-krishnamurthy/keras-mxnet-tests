#!/bin/sh
set -e

# Install prerequisites
pip install memory_profiler --user
pip install psutil --user
# Dependencies required for Keras installation
pip install pyyaml --user

pip install --upgrade pip --user
pip install --upgrade six --user

# For model load and save functionality
# sudo apt-get install -y python-h5py

/usr/bin/nvidia-smi

# Install MXNet
cd mxnet
make clean
make -j $(nproc) USE_OPENCV=1 USE_BLAS=atlas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
cd python
python setup.py install --user

# Install Keras with MXNet backend
cd ../../keras/
python setup.py install --user

########## Set Environment Variables ########
cd ../keras-mxnet-tests/keras1.2/
echo "Setting Environment Variables for MXNet Keras Integration Tests on CPU machine"

export KERAS_BACKEND="mxnet"
export MXNET_KERAS_TEST_MACHINE="GPU"

########## Call the test script with 1 GPUS ############
export GPU_NUM="1"
echo "Running MXNet Keras Integration Test on GPU machine with 1 GPUs"
nosetests --with-xunit --quiet --nologcapture nightly_test/

########## Run Keras unit tests ########################

cd ../../keras/
# Prerequisites For running unit tests
pip install pytest --user
pip install -e .[tests] --user
pip install image --user
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp27-none-linux_x86_64.whl --user

echo "Running Keras unit tests"
python -m py.test tests/integration_tests/
python -m py.test tests/keras/
