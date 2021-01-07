## CoScal

(Abstract) Recent mobile devices, with powerful GPUs, are capable of running highly computation and memory expensive neural network tasks like continuous vision workloads.
However, high power consumption of computing neural networks is a major concern in mobile continuous vision applications.
DVFS is an efficient power saving technique that can be used to lower power consumption of computing neural networks on mobile devices.
We analyze the effect of DVFS on a system by roofline analysis and devise a novel analytic model: DVFS-incorporated roofline model.
Based on the analytic model, we create CoScal, a frequency scaling system which uses co-scaling of both GPU core and bus frequency levels to reduce power consumption.
CoScal automatically tests the performance when running a neural network inference on a mobile device at certain GPU core and bus frequencies and finds the optimal GPU core and bus frequencies for a given performance target.
We test CoScal on Pixel 3a by measuring the total power consumption of the device when running a continuous vision task.
Experimental results show that CoScal is able to steadily save more power compared to using default DVFS governors.

We provide a guide on how to build and use this program here.

## 1. Philipp Wollermann's guide on how to build Tensorflow with Bazelisk (modified to fit for Android)
Source: https://gist.github.com/philwo/f3a8144e46168f23e40f291ffe92e63c

```bash
$ sudo apt install curl

# Install Bazelisk. (Versions can be different, I used 1.7.4 for this program)
$ sudo curl -Lo /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.7.4/bazelisk-linux-amd64
$ sudo chmod +x /usr/local/bin/bazel

# This should work and print a Bazelisk and Bazel version.
$ bazel version
Bazelisk version: v1.7.4
Build label: 3.1.0
[...]

# Now we're following the official "Build from source" steps:
# https://www.tensorflow.org/install/source
$ sudo apt install python python3-{dev,pip,six,numpy,wheel,setuptools,mock}
$ pip3 install -U --user 'future>=0.17.1'
$ pip3 install -U --user keras_applications --no-deps
$ pip3 install -U --user keras_preprocessing --no-deps

# Download TensorFlow 2.3.0:
$ git clone https://github.com/tensorflow/tensorflow.git --branch v2.3.0
$ cd tensorflow

# Configure for Android
$ ./configure
# You can configure all the settings to default by pressing Enter but should answer y to this specific question:
Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: y

# Then answer these accordingly
Searching for NDK and SDK installations.

Please specify the home path of the Android NDK to use. [Default is /home/jongminkim/Android/Sdk/ndk-bundle]: /home/jongminkim/Android/Sdk/ndk/21.3.6528147
WARNING: The NDK version in /home/jongminkim/Android/Sdk/ndk/21.3.6528147 is 21, which is not supported by Bazel (officially supported versions: [10, 11, 12, 13, 14, 15, 16, 17, 18]). Please use another version. Compiling Android targets may result in confusing errors.

Please specify the (min) Android NDK API level to use. [Available levels: ['16', '17', '18', '19', '21', '22', '23', '24', '26', '27', '28', '29', '30']] [Default is 21]: 21

Please specify the home path of the Android SDK to use. [Default is /home/jongminkim/Android/Sdk]: /home/jongminkim/Android/Sdk

Please specify the Android SDK API level to use. [Available levels: ['26', '28', '29', '30']] [Default is 30]: 26

Please specify an Android build tools version to use. [Available versions: ['28.0.3', '29.0.2', '30.0.1']] [Default is 30.0.1]: 30.0.1
# The warning can be ignored, since NDK version 21 is required for proper operation
# You can use higher Android SDK API level if you'd like for more recent Android versions
# It is convenient to use Android Studio by Google to manage SDKs and NDKs
# Find the guides there for more information
```


## 2. How to build and use
```bash
# Move the files in @dvfs_project/src to @tensorflow_source/tensorflow/lite/delegates/gpu/cl/testing
# path doesn't really matter but you have to edit the BUILD file in @dvfs_project/src for doing so
$ cp ./dvfs_project/src/* ./tensorflow/tensorflow/lite/delegates/gpu/cl/testing

# Build using Bazel
$ cd tensorflow
$ bazel build -c opt --cxxopt=--std=c++11 --config=android_arm64 //tensorflow/lite/delegates/gpu/cl/testing:dvfs

# Then push the results to the device in a convenient folder
# "/data/local/tmp" is always a good option
$ adb push bazel-bin/tensorflow/lite/delegates/gpu/cl/testing/dvfs /data/local/tmp

# Then using adb, run the program in su
# arguments are tflite file name (in the same folder) and target latency in ms
# For example:
$ adb shell
adb:/ $ su
adb:/ # cd /data/local/tmp
adb:/data/local/tmp # ./dvfs mobilenet_v2.tflite 16.666
```
