#!/bin/bash
# This script is used to change some configurations of GPU power management via sysfs interface.
# This code is hardware-specific and cannot be applied to mobile SoCs other than Sony Xperia X Performance (F8131).
# It changes the configurations of the kernel of Android 8.0.0 Oreo for Adreno 530 in Snapdragon 820.
# Plus runs some benchmark tests for possible GPU frequencies.
# Based on Tensorflow Lite Benchmark Tool

PROJECT_HOME="$(pwd)/.."
SYSFS_DIR="/sys/class/kgsl/kgsl-3d0"
GRAPH_NAME="mobilenet_v2_1.0_224.tflite"
# TODO: get user input for the GRAPH_NAME
HOST_EXE="$PROJECT_HOME/tools/benchmark_model"
HOST_GRAPH="$PROJECT_HOME/models/$GRAPH_NAME"
BENCHMARK_DIR="/data/local/tmp"
BENCHMARK_EXE="$BENCHMARK_DIR/benchmark_model"
BENCHMARK_GRAPH="$BENCHMARK_DIR/$GRAPH_NAME"
BENCHMARK_OPT="--use_gpu=true --use_fp16=true --num_runs=100"
BENCHMARK_POST="| grep avg"
BENCHMARK="$BENCHMARK_EXE --graph=$BENCHMARK_GRAPH $BENCHMARK_OPT $BENCHMARK_POST"
# Do not allow any error


# needs a rooted phone
adb root

if [ ! $(adb shell "su -c '[ -e $BENCHMARK_EXE ] && echo 1'") ]; then
    adb push $HOST_EXE $BENCHMARK_EXE
fi

if [ ! $(adb shell "su -c '[ -e $BENCHMARK_GRAPH ] && echo 1'") ]; then
    adb push $HOST_GRAPH $BENCHMARK_GRAPH
fi

adb shell "su -c 'echo 1 > $SYSFS_DIR/force_clk_on'"
adb shell "su -c 'echo 10000000 > $SYSFS_DIR/idle_timer'"
adb shell "su -c 'echo performance > $SYSFS_DIR/devfreq/governor'"
# userspace governor not working
# adb shell "su -c 'echo userspace > $SYSFS_DIR/devfreq/governor'"
IFS=' ' read -r -a freq_array <<< $(adb shell "su -c 'cat /sys/class/kgsl/kgsl-3d0/gpu_available_frequencies'")
iter=1
for freq in "${freq_array[@]}"
do
    adb shell "su -c 'echo $freq > $SYSFS_DIR/max_gpuclk'"
    adb shell "su -c 'echo $freq > $SYSFS_DIR/gpuclk'"
    echo "---Test $iter for frequency $freq---"
    adb shell "su -c '$BENCHMARK'"
    let iter=$iter+1
done

