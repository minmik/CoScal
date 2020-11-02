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
EXE_NAME="mixbench-ocl-ro"
HOST_EXE="$PROJECT_HOME/tools/$EXE_NAME"
HOST_GRAPH="$PROJECT_HOME/models/$GRAPH_NAME"
BENCHMARK_DIR="/data/local/tmp"
BENCHMARK_EXE="$BENCHMARK_DIR/$EXE_NAME"
#BENCHMARK_GRAPH="$BENCHMARK_DIR/$GRAPH_NAME"
#BENCHMARK_OPT="--use_gpu=true --use_fp16=true --num_runs=100"
#BENCHMARK_POST="| grep avg"
BENCHMARK="$BENCHMARK_EXE $BENCHMARK_GRAPH $BENCHMARK_OPT $BENCHMARK_POST"
# Do not allow any error


# needs a rooted phone
adb root

if [ ! $(adb shell "su -c '[ -e $BENCHMARK_EXE ] && echo 1'") ]; then
    adb push $HOST_EXE $BENCHMARK_EXE
	adb shell "su -c 'chmod 777 > $BENCHMARK_EXE'"
fi

if [ ! $(adb shell "su -c '[ -e $BENCHMARK_GRAPH ] && echo 1'") ]; then
    adb push $HOST_GRAPH $BENCHMARK_GRAPH
fi

#fix gpubw to max
#adb shell "su -c 'cd /sys/class/devfreq/soc\:qcom,gpubw && echo userspace > governor'"
adb shell "su -c 'echo 1 > /sys/devices/system/cpu/cpu0/online'"
adb shell "su -c 'echo 1 > /sys/devices/system/cpu/cpu1/online'"
adb shell "su -c 'echo 1 > /sys/devices/system/cpu/cpu2/online'"
adb shell "su -c 'echo 1 > /sys/devices/system/cpu/cpu3/online'"
adb shell "su -c 'stop thermald'"
adb shell "su -c 'stop mpdecision'"

adb shell "su -c 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'"
adb shell "su -c 'echo performance > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor'"
adb shell "su -c 'echo performance > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor'"
adb shell "su -c 'echo performance > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor'"

adb shell "su -c 'echo 0 > $SYSFS_DIR/bus_split'"
adb shell "su -c 'echo 1 > $SYSFS_DIR/force_bus_on'"
adb shell "su -c 'echo 1 > $SYSFS_DIR/force_clk_on'"
adb shell "su -c 'echo 10000000 > $SYSFS_DIR/idle_timer'"
#adb shell "su -c 'echo performance > $SYSFS_DIR/devfreq/governor'"
# userspace governor not working
# adb shell "su -c 'echo userspace > $SYSFS_DIR/devfreq/governor'"
IFS=' ' read -r -a freq_array <<< $(adb shell "su -c 'cat /sys/class/kgsl/kgsl-3d0/gpu_available_frequencies'")
IFS=' ' read -r -a bw_array <<< $(adb shell "su -c 'cat /sys/class/devfreq/soc\:qcom,gpubw/available_frequencies'")
freq=624000000
bw=13763
pwrlevel=0
adb shell "su -c 'echo $bw > /sys/class/devfreq/soc\:qcom,gpubw/min_freq'"
adb shell "su -c 'echo $bw > /sys/class/devfreq/soc\:qcom,gpubw/max_freq'"
adb shell "su -c 'echo $bw > /sys/class/devfreq/soc\:qcom,gpubw/min_freq'"
adb shell "su -c 'echo $freq > $SYSFS_DIR/devfreq/min_freq'"
adb shell "su -c 'echo $freq > $SYSFS_DIR/devfreq/max_freq'"
adb shell "su -c 'echo $freq > $SYSFS_DIR/devfreq/min_freq'"
adb shell "su -c 'echo $pwrlevel > $SYSFS_DIR/min_pwrlevel'"
adb shell "su -c 'echo $pwrlevel > $SYSFS_DIR/max_pwrlevel'"
adb shell "su -c 'echo $pwrlevel > $SYSFS_DIR/min_pwrlevel'"
echo "---Test $iter for frequency $freq and bw $bw---"
adb shell "su -c 'sleep 1'"
adb shell "su -c '$BENCHMARK'"

