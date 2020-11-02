/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Modified by Jongmin Kim (kimcm3310@gmail.com)
/* Copyright 2020 Jongmin Kim. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "dvfs.h"

namespace tflite {
namespace gpu {
namespace cl {

int write_to_file(const std::string path, const std::string value) {
    int result = 0;
    std::ofstream writeFile(path);
    if(writeFile.is_open())
        writeFile << value;
    else
        result = -1;
    writeFile.close();
    return result;
}

GPUExecution::GPUExecution(const std::string _model_name, double target) {
    model_name = _model_name;
    currentCoreFreqIndex = -1;
    currentBusFreqIndex = -1;
    executionTime = 0.0;
    latencyTarget = target;
    
    /* Maybe will be used in mixbench
    opIntensity = 4.75;
    coreBenchResults = new Performance[numCoreFreq];
    busBenchResults = new Performance[numBusFreq];
    */

    coreBenchResults_ms = new double[numCoreFreq];
    busBenchResults_ms = new double[numBusFreq];

    // Set some environments
    flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
    EXIT_IF_ERROR(BuildFromFlatBuffer(*flatbuffer, op_resolver, &graph_cl));
    
    EXIT_IF_ERROR(CreateEnvironment(&env));
    create_info.precision = env.IsSupported(CalculationsPrecision::F16)
                                ? CalculationsPrecision::F16
                                : CalculationsPrecision::F32;
    create_info.storage_type = GetFastestStorageType(env.device());
    std::cout << "Precision: " << ToString(create_info.precision) << std::endl;
    std::cout << "Storage type: " << ToString(create_info.storage_type)
            << std::endl;
    EXIT_IF_ERROR(
        context.InitFromGraphWithTransforms(create_info, &graph_cl, &env));


    if(set_gpu_performance_mode() == -1) {
        std::cerr << "Something went wrong when setting GPU to performance mode"
                << std::endl << "Check permission" << std::endl;
        exit(-1);
    }
    if(set_gpu_to_max() == -1) {
        std::cerr << "Something went wrong when setting GPU to level " 
                << "0, 0" << std::endl
                << "Check permission" << std::endl;
        exit(-1);
    }

    EXIT_IF_ERROR(RunModelSample());

    // Latency Target Lowering when target unachievable
    if(currentCoreFreqIndex == 0 && currentBusFreqIndex == 0 && executionTime > latencyTarget) {
        double oldTarget = latencyTarget;
        while(executionTime > latencyTarget) {
            latencyTarget *= 2.0f;
        }
        std::cerr << "Latency Target " << oldTarget << " unachievable, " <<
                    "lowering the latency target to " << latencyTarget << std::endl;
    }
}


int GPUExecution::set_gpu_performance_mode(void) {
    int result = -1;
#ifdef F8131
    result = 0;
    result += write_to_file(CPU_SYSFS_PATH + "/cpu0/online", "1");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu0/cpufreq/scaling_governor", "performance");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu1/online", "1");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu1/cpufreq/scaling_governor", "performance");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu2/online", "1");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu2/cpufreq/scaling_governor", "performance");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu3/online", "1");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu3/cpufreq/scaling_governor", "performance");
    result += write_to_file(GPU_SYSFS_PATH + "/bus_split", "0");
    result += write_to_file(GPU_SYSFS_PATH + "/force_bus_on", "1");
    result += write_to_file(GPU_SYSFS_PATH + "/force_clk_on", "1");
    result += write_to_file(GPU_SYSFS_PATH + "/force_rail_on", "1");
    result += write_to_file(GPU_SYSFS_PATH + "/idle_timer", "10000000");
    result += write_to_file(GPU_SYSFS_PATH + "/devfreq/governor", "performance");
    result += write_to_file(GPU_GPUBW_PATH + "/governor", "performance");
#endif
#ifdef FLAME
    result = 0;
    result += write_to_file(CPU_SYSFS_PATH + "/cpu0/online", "1");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu0/cpufreq/scaling_governor", "performance");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu1/online", "1");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu1/cpufreq/scaling_governor", "performance");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu2/online", "1");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu2/cpufreq/scaling_governor", "performance");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu3/online", "1");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu3/cpufreq/scaling_governor", "performance");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu4/online", "1");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu4/cpufreq/scaling_governor", "performance");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu5/online", "1");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu5/cpufreq/scaling_governor", "performance");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu6/online", "1");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu6/cpufreq/scaling_governor", "performance");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu7/online", "1");
    result += write_to_file(CPU_SYSFS_PATH + "/cpu7/cpufreq/scaling_governor", "performance");
    result += write_to_file(GPU_SYSFS_PATH + "/bus_split", "0");
    result += write_to_file(GPU_SYSFS_PATH + "/force_bus_on", "1");
    result += write_to_file(GPU_SYSFS_PATH + "/force_clk_on", "1");
    result += write_to_file(GPU_SYSFS_PATH + "/force_rail_on", "1");
    result += write_to_file(GPU_SYSFS_PATH + "/idle_timer", "10000000");
    result += write_to_file(GPU_SYSFS_PATH + "/devfreq/governor", "performance");
    result += write_to_file(GPU_GPUBW_PATH + "/governor", "performance");
#endif
    return result < 0 ? -1 : 0; 
}

inline int GPUExecution::set_gpu_to_max(void) {
    return set_gpu_to_level(0, 0);
}

int GPUExecution::set_gpu_to_level(int core_index, int bus_index) {
    //check for index validity
    if(core_index < 0 || core_index >= numCoreFreq) {
        std::cerr << "set_gpu_to_level: wrong core_index "
                    << core_index << " specified" << std::endl;
        return -1;
    }
    if(bus_index < 0 || bus_index >= numBusFreq) {
        std::cerr << "set_gpu_to_level: wrong bus_index"
                    << bus_index << " specified" << std::endl;
        return -1;
    }

    int result = -1;
#if defined(F8131) || defined(FLAME)
    result = 0;
    
    // Scaling core frequency
    result += write_to_file(GPU_SYSFS_PATH + "/max_gpuclk", coreFreq[core_index]);
    if(result >= 0)
        currentCoreFreqIndex = core_index;

    // Scaling bus frequency
    const std::string targetBusFreq = busFreq[bus_index];
    // leveling down the bus frequency (higher index means lower freq)
    if(currentBusFreqIndex < bus_index) {
        result += write_to_file(GPU_GPUBW_PATH + "/min_freq", targetBusFreq);
        result += write_to_file(GPU_GPUBW_PATH + "/max_freq", targetBusFreq);
    }
    // leveling up the bus frequency (higher index means lower freq)
    else {
        result += write_to_file(GPU_GPUBW_PATH + "/max_freq", targetBusFreq);
        result += write_to_file(GPU_GPUBW_PATH + "/min_freq", targetBusFreq);
    }
    if(result >= 0)
        currentBusFreqIndex = bus_index;
#endif
    return result < 0 ? -1 : 0;
}



void GPUExecution::perform_benchmark(void) {
    std::cout << "Performing Benchmark..." << std::endl;
    for(int i = 0; i < numCoreFreq; i++) {
        if(set_gpu_to_level(i, 0) == -1) {
            std::cerr << "Something went wrong when setting GPU to level " 
                    << i << ", 0" << std::endl
                    << "Check permission" << std::endl;
            exit(-1);
        }
        EXIT_IF_ERROR(RunModelSample());
        coreBenchResults_ms[i] = executionTime;
    }

    for(int j = 0; j < numBusFreq; j++) {
        if(set_gpu_to_level(0, j) == -1) {
            std::cerr << "Something went wrong when setting GPU to level "
                    << "0, " << j << std::endl
                    << "Check permission" << std::endl;
            exit(-1);
        }
        EXIT_IF_ERROR(RunModelSample());
        busBenchResults_ms[j] = executionTime;
    }

    if(set_gpu_to_max() == -1) {
        std::cerr << "Something went wrong when setting GPU to level " 
                << "0, 0" << std::endl
                << "Check permission" << std::endl;
        exit(-1);
    }
    std::cout << "Benchmarking Done!" << std::endl;
}


void GPUExecution::set_dvfs_using_bench(void) {
    int i, j;
    for(i = 1; i < numCoreFreq; i++) {
        if(coreBenchResults_ms[i] > latencyTarget)
            break;
    }
    for(j = 1; j < numBusFreq; j++) {
        if(busBenchResults_ms[j] > latencyTarget)
            break;
    }
    if(set_gpu_to_level(i - 1, j - 1) == -1) {
        std::cerr << "Something went wrong when setting GPU to level " 
                << i - 1 << ", " << j - 1 << std::endl 
                << "Check permission" << std::endl;
        exit(-1);
    }
    std::cout << "DVFS set to CORE: " << coreFreq[i - 1]
            << " and BUS: " << busFreq[j - 1] 
            << " according to benchmark results!" << std::endl;
}



void GPUExecution::fine_tune(void) {
    EXIT_IF_ERROR(RunModelSample());
    while(true) {
        int check_repeat = 0;
        while(check_repeat < 10 && executionTime < latencyTarget) {
            EXIT_IF_ERROR(RunModelSample());
            check_repeat ++;
        }
        if(executionTime < latencyTarget)
            break;


        // scale up the frequency setting that is causing the bottleneck

        // 1. When core is already max
        if (currentCoreFreqIndex == 0) {
            // 1-1. When bus is also max
            if (currentBusFreqIndex == 0) {
                std::cerr << "Latency Target " << latencyTarget << " unachievable, " <<
                            "execution time now: " << executionTime << std::endl;
                std::cerr << "fine tuning failed" << std::endl;
                exit(-1);
            }
            // 1-2. When bus is tunable
            else if (currentBusFreqIndex > 0 && currentBusFreqIndex < numBusFreq) {
                if(set_gpu_to_level(currentCoreFreqIndex, currentBusFreqIndex - 1) == -1) {
                    std::cerr << "Something went wrong when setting GPU to level "
                    << currentCoreFreqIndex<< ", " << currentBusFreqIndex - 1<< std::endl 
                    << "Check permission" << std::endl;
                    exit(-1);
                }
            }
            // 1-3. Error
            else {
                std::cerr << "FATAL ERROR: Wrong bus freq index while fine tuning!!!" << std::endl;
                exit(-1);
            }
        }

        // 2. When core is tunable
        else if(currentCoreFreqIndex > 0 && currentCoreFreqIndex < numCoreFreq) {
            // 2-1. When bus is max
            if(currentBusFreqIndex == 0) {
                if(set_gpu_to_level(currentCoreFreqIndex  - 1, currentBusFreqIndex) == -1){
                    std::cerr << "Something went wrong when setting GPU to level "
                    << currentCoreFreqIndex - 1<< ", " << currentBusFreqIndex<< std::endl 
                    << "Check permission" << std::endl;
                    exit(-1);
                }
            }
            // 2-2. When bus is also tunable
            else if(currentBusFreqIndex > 0 && currentBusFreqIndex < numBusFreq) {
                //core is being the bottleneck
                if(coreBenchResults_ms[currentCoreFreqIndex] >= busBenchResults_ms[currentBusFreqIndex]) {
                    if(set_gpu_to_level(currentCoreFreqIndex  - 1, currentBusFreqIndex) == -1){
                        std::cerr << "Something went wrong when setting GPU to level "
                                << currentCoreFreqIndex - 1<< ", " << currentBusFreqIndex<< std::endl 
                                << "Check permission" << std::endl;
                        exit(-1);
                    }
                }
                //bus is being the bottleneck
                else {
                    if(set_gpu_to_level(currentCoreFreqIndex, currentBusFreqIndex - 1) == -1) {
                        std::cerr << "Something went wrong when setting GPU to level "
                                << currentCoreFreqIndex<< ", " << currentBusFreqIndex - 1<< std::endl 
                                << "Check permission" << std::endl;
                        exit(-1);
                    }
                }
            }
            // 2-3. Error
            else {
                std::cerr << "FATAL ERROR: Wrong bus freq index while fine tuning!!!" << std::endl;
                exit(-1);
            }
        }

        // 3. Error
        else {
            std::cerr << "FATAL ERROR: Wrong core freq index while fine tuning!!!" << std::endl;
            exit(-1);
        }

        // recompute executionTime
        EXIT_IF_ERROR(RunModelSample());
    }
    std::cout<< "Fine tuning done: core " << coreFreq[currentCoreFreqIndex] <<
                " bus " << busFreq[currentBusFreqIndex] << std::endl;
}


// Run tflite model, copyright: Tensorflow Authors
// Original Code: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/cl/testing/performance_profiling.cc
// modified by Jongmin Kim
absl::Status GPUExecution::RunModelSample(void) {
    auto* queue = env.profiling_queue();
    ProfilingInfo profiling_info;
    RETURN_IF_ERROR(context.Profile(queue, &profiling_info));
    //std::cout << profiling_info.GetDetailedReport() << std::endl;
    //uint64_t mem_bytes = context.GetSizeOfMemoryAllocatedForIntermediateTensors();
    //std::cout << "Memory for intermediate tensors - "
    //          << mem_bytes / 1024.0 / 1024.0 << " MB" << std::endl;

    const int num_runs_per_sec = std::max(
        1, static_cast<int>(1000.0f / absl::ToDoubleMilliseconds(
                                        profiling_info.GetTotalTime())));


    const auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < num_runs_per_sec; ++k) {
        EXIT_IF_ERROR(context.AddToQueue(env.queue()));
    }
    RETURN_IF_ERROR(env.queue()->WaitForCompletion());
    const auto end = std::chrono::high_resolution_clock::now();


    const double total_time_ms = (end - start).count() * 1e-6f;
    double average_inference_time = total_time_ms / num_runs_per_sec;

    std::cout << "Measured Latency - " << average_inference_time << "ms" << std::endl;
    executionTime = average_inference_time;

    return absl::OkStatus();
}


// Run tflite model periodically, copyright: Tensorflow Authors
// Original Code: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/cl/testing/performance_profiling.cc
// modified by Jongmin Kim
void RunPeriodically(const std::string model_name, double period) {
    // Some Environment Setting
    auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
    GraphFloat32 graph_cl;
    ops::builtin::BuiltinOpResolver op_resolver;
    EXIT_IF_ERROR(BuildFromFlatBuffer(*flatbuffer, op_resolver, &graph_cl));

    Environment env;
    EXIT_IF_ERROR(CreateEnvironment(&env));

    InferenceContext::CreateInferenceInfo create_info;
    create_info.precision = env.IsSupported(CalculationsPrecision::F16)
                                ? CalculationsPrecision::F16
                                : CalculationsPrecision::F32;
    create_info.storage_type = GetFastestStorageType(env.device());
    std::cout << "Precision: " << ToString(create_info.precision) << std::endl;
    std::cout << "Storage type: " << ToString(create_info.storage_type)
            << std::endl;
    InferenceContext context;
    EXIT_IF_ERROR(
        context.InitFromGraphWithTransforms(create_info, &graph_cl, &env));

    //const int kNumRuns = 10;

    struct timespec ts;
    int err = clock_getres(CLOCK_MONOTONIC, &ts);

    std::cout << "Timer starting. System resolution: "
            << ts.tv_nsec << "  ns" << std::endl;
    
    err = clock_gettime(CLOCK_MONOTONIC, &ts);
    while (true) {
        long next_tick = (ts.tv_sec * 1000000000L + ts.tv_nsec) + (long) (period * 1000000L);
        ts.tv_sec = next_tick / 1000000000L;
        ts.tv_nsec = next_tick % 1000000000L;
        std::cout << (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0 << std::endl;
        EXIT_IF_ERROR(context.AddToQueue(env.queue()));
        err = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);
        EXIT_IF_ERROR(env.queue()->WaitForCompletion());
    }
}


} // namespace cl
} // namespace gpu
} // namespace tflite

int main(int argc, char** argv) {
    double target_latency_ms;

    if (argc <= 1) {
        std::cerr << "Expected model path as second argument.";
        return -1;
    }
    if (argc <= 2) {
        std::cerr << "Error: Target performance not specified" << std::endl;
        return -1;
    }
    else if (argc <= 3){
        target_latency_ms = std::atof(argv[2]);
        auto load_status = tflite::gpu::cl::LoadOpenCL();
        if (!load_status.ok()) {
            std::cerr << load_status.message();
            return -1;
        }
        std::cout << "----------------------------------------------" << std::endl;
        std::cout << "Starting DVFS adjustment..." << std::endl;
        std::cout << "Model: " << argv[1] << std::endl;
        std::cout << "Target Latency: " << target_latency_ms << " ms" << std::endl;
        std::cout << "----------------------------------------------" << std::endl;
        // Creates GPUExecution instance
        // Notice: the constructor does a lot of job
        tflite::gpu::cl::GPUExecution execution(argv[1], target_latency_ms);


        // TODO
        // benchmarking
        execution.perform_benchmark();
        execution.set_dvfs_using_bench();

        // should be automatically decided by benchmarking results
        execution.fine_tune();
        return 0;
    }
    else if(argv[3]){
        target_latency_ms = std::atof(argv[2]);
        auto load_status = tflite::gpu::cl::LoadOpenCL();
        if (!load_status.ok()) {
            std::cerr << load_status.message();
            return -1;
        }
        tflite::gpu::cl::RunPeriodically(argv[1], target_latency_ms);
        return -1; // This program does not stop unless aborted
    }

}
