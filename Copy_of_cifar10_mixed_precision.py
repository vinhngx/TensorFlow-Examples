# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"colab": {}, "colab_type": "code", "id": "tuOe1ymfHZPu"}
# Copyright 2019 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# + {"colab_type": "text", "id": "MfBg1C5NB3X0", "cell_type": "markdown"}
# <img src="https://upload.wikimedia.org/wikipedia/en/thumb/6/6d/Nvidia_image_logo.svg/200px-Nvidia_image_logo.svg.png" width="90px" align="right" style="margin-right: 0px;">
#
# # Mixed Precision Training of CNN

# + {"colab_type": "text", "id": "xHxb-dlhMIzW", "cell_type": "markdown"}
# ## Overview
#
# In this example, we will speed-up the training of a simple CNN with mixed precision to perform image classification on the CIFAR10 dataset.
#
# By using mixed precision, we can reduce the training time without a significant impact on classification accuracy. For example, using the NVIDIA Tesla T4 GPU on Google Colab, we can reduce the training time (using the same model and batch size) over 10 epochs from about 600 seconds (FP32) to about 300 seconds with mixed precision, without sacrificing classification accuracy.
#
# **How mixed precision works**
#
# **Mixed precision** is the use of both float16 and float32 data types when training a model.
#
# Performing arithmetic operations in float16 takes advantage of the performance gains of using lower precision hardware (such as Tensor Cores). Due to the smaller representable range of float16, performing the entire training with float16 tensors can result in underflow and overflow errors.
#
# However, *performing only certain arithmetic operations* in float16 results in performance gains when using compatible hardware accelerators, decreasing training time and reducing memory usage, typically without sacrificing model performance.
#
# To learn more about mixed precision and how it works:
#
# * [Overview of Automatic Mixed Precision for Deep Learning](https://developer.nvidia.com/automatic-mixed-precision)
# * [NVIDIA Mixed Precision Training Documentation](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)
# * [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html)
# * [Information about NVIDIA Tensor Cores](https://developer.nvidia.com/tensor-cores)
# * [Post on TensorFlow blog explaining Automatic Mixed Precision](https://medium.com/tensorflow/automatic-mixed-precision-in-tensorflow-for-faster-ai-training-on-nvidia-gpus-6033234b2540)
#
# Do note that some of the resources above reference may an older version of the mixed precision API (setting the `TF_ENABLE_AUTO_MIXED_PRECISION` environment variable). The method presented in this notebook is the current API used in TensorFlow 1.14 and newer.

# + {"colab_type": "text", "id": "MUXex9ctTuDB", "cell_type": "markdown"}
# ## Setup and Requirements

# + {"colab_type": "text", "id": "0yh5tPwanSjf", "cell_type": "markdown"}
# **Hardware requirements**
#
# * NVIDIA Tensor Core GPU (Compute Capability >= `7.0`)
#
# **Software requirements**
#
# * TensorFlow version >= `1.14.0-rc0`
#
# The following section will import the necessary libraries and check if these requirements are met.

# + {"colab": {}, "colab_type": "code", "id": "IqR2PQG4ZaZ0"}
import time
import numpy as np

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 53}, "colab_type": "code", "id": "SyOV7ojinSjf", "outputId": "92f95198-c4d5-4eff-a855-b4af36f2a392"}
from tensorflow.python.client import device_lib

def check_tensor_core_gpu_present():
    local_device_protos = device_lib.list_local_devices()
    for line in local_device_protos:
        if "compute capability" in str(line):
            compute_capability = float(line.physical_device_desc.split("compute capability: ")[-1])
            if compute_capability>=7.0:
                return True

print("TensorFlow version is", tf.__version__)

try:
    # check and assert TensorFlow >= 1.14
    tf_version_list = tf.__version__.split(".")
    if int(tf_version_list[0]) < 2:
        assert int(tf_version_list[1]) >= 14
except:
    print("TensorFlow 1.14.0 or newer is required.")
    
print("Tensor Core GPU Present:", check_tensor_core_gpu_present())
if check_tensor_core_gpu_present():
    pass
else:
    !nvidia-smi
    assert check_tensor_core_gpu_present() == True

# + {"colab_type": "text", "id": "QKp40qS-DGEZ", "cell_type": "markdown"}
# ## Import the Dataset
#
# Import the CIFAR10 image dataset from `tf.keras.datasets`

# + {"colab": {}, "colab_type": "code", "id": "GQq1V2EmnSjj"}
# The data, split between train and test sets

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

num_classes = np.max(y_train) + 1

# Convert class vectors to binary class matrices

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# + {"colab_type": "text", "id": "EPSlI16QnSjl", "cell_type": "markdown"}
# Preprocess the images by scaling the values from the range `0 ~ 255` to the range `0 ~ 1`

# + {"colab": {}, "colab_type": "code", "id": "4W52Ac5-nSjl"}
def normalize(ndarray):
    ndarray = ndarray.astype("float32")
    ndarray = ndarray/255.0
    return ndarray

x_train = normalize(x_train)
x_test = normalize(x_test)


# + {"colab_type": "text", "id": "0zTEfgJFnSjn", "cell_type": "markdown"}
# ## Define the Model
#
# Define a reusable helper function to return a simple CNN

# + {"colab": {}, "colab_type": "code", "id": "edD6HTTmnSjo"}
def create_model(num_classes=10):
    """
    Returns a simple CNN suitable for classifiying images from CIFAR10
    """
    # model parameters
    act = "relu"
    pad = "same"
    ini = "he_uniform"
    
    model = tf.keras.models.Sequential([
        Conv2D(128, (3, 3), activation=act, padding=pad, kernel_initializer=ini,
               input_shape=(32,32,3)),
        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(256, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
        Conv2D(128, (3, 3), activation=act, padding=pad, kernel_initializer=ini),
        MaxPooling2D(pool_size=(4,4)),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dropout(rate=0.2),
        Dense(num_classes, activation="softmax")
    ])

    return model


# + {"colab": {"base_uri": "https://localhost:8080/", "height": 737}, "colab_type": "code", "id": "VsZy8zzwjz1V", "outputId": "c1b39fc5-976e-4606-a3a5-993dcd49236f"}
model = create_model(num_classes)
model.summary()

# + {"colab_type": "text", "id": "OB8kdGNTnSjr", "cell_type": "markdown"}
# ## Training the Model
#
# Train and benchmark the same model trained with and without mixed precision

# + {"colab": {}, "colab_type": "code", "id": "vXHS_e8lnSjs"}
# training parameters
BATCH_SIZE = 320
N_EPOCHS = 10
opt = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.5)


# + {"colab": {}, "colab_type": "code", "id": "G05htP5vnSju"}
def train_model(mixed_precision, optimizer):
    """
    Trains a CNN to classify images on CIFAR10,
    and returns the training and classification performance
    
    Args:
        mixed_precision: `True` or `False`
        optimizer: An instance of `tf.keras.optimizers.Optimizer`
    """
    model = create_model(num_classes)

    if mixed_precision:
        import tensorflow
        optimizer = tensorflow.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    
    train_start = time.time()

    train_log = model.fit(x_train, y_train,
                          batch_size=BATCH_SIZE,
                          epochs=N_EPOCHS,
                          use_multiprocessing=True,
                          workers=2)

    train_end = time.time()

    score = model.evaluate(x_test, y_test)
    
    results = {"test_loss": score[0],
               "test_acc": score[1],
               "train_time": train_end-train_start,
               "train_log": train_log}
    
    return results


# + {"colab_type": "text", "id": "B1o7YirynSjw", "cell_type": "markdown"}
# ### Training without Mixed Precision

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 577}, "colab_type": "code", "id": "jS4XPcLFnSjw", "outputId": "e07748c6-1d27-4a0c-fcc4-36fd6222b26d"}
fp32_results = train_model(mixed_precision=False, optimizer=opt)

test_acc = round(fp32_results["test_acc"]*100, 1)
train_time = round(fp32_results["train_time"], 1)

print(test_acc, "% achieved in", train_time, "seconds")

# + {"colab": {}, "colab_type": "code", "id": "QiRUF-v-omKO"}
# to ensure accuracy of timing benchmark
# we give the GPU 10 seconds to cool down

tf.keras.backend.clear_session()

time.sleep(10)

# + {"colab_type": "text", "id": "lnA6jdbVnSjy", "cell_type": "markdown"}
# ### Training with Mixed Precision

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 431}, "colab_type": "code", "id": "KtylpxOmceaC", "outputId": "8a0c0a05-d5ad-40bd-b95d-bd77f2c1daf1"}
mp_results = train_model(mixed_precision=True, optimizer=opt)

test_acc = round(mp_results["test_acc"]*100, 1)
train_time = round(mp_results["train_time"], 1)

print(test_acc, "% achieved in", train_time, "seconds")

# + {"colab_type": "text", "id": "1BR_XJP9nSj0", "cell_type": "markdown"}
# ### Evaluate the Model Performance

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 295}, "colab_type": "code", "id": "mMOeXVmbdilM", "outputId": "bcd95df7-5976-4d99-e59d-8b3fed24b74e"}
import matplotlib.pyplot as plt
# %matplotlib inline

plt.plot(fp32_results["train_log"].history["loss"], label="FP32")
plt.plot(mp_results["train_log"].history["loss"], label="Mixed Precision")
plt.title("Performance Comparison")
plt.ylabel("Training Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 35}, "colab_type": "code", "id": "kETVyzl4tJhv", "outputId": "f3032545-06da-4f41-b673-e4f36f767c4d"}
speed_up = round(100 * fp32_results["train_time"]/mp_results["train_time"], 1)

print("Total speed-up:", speed_up, "%")

# + {"colab_type": "text", "id": "78HBT9cQXJko", "cell_type": "markdown"}
# ## Conclusions
#
# * Mixed Precision training provides a significant speed-up over FP32 (single-precision) training
# * Switch to using mixed precision by wrapping a `tf.keras.optimizers` Optimizer in `tf.train.experimental.enable_mixed_precision_graph_rewrite()`
#
