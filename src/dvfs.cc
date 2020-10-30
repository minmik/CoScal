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

int GPUExecution::set_gpu_performance_mode(void) {
    int result = -1;
#if defined(F8131) || defined(FLAME)
    result = 0;
    result += write_to_file(GPU_SYSFS_PATH + "/bus_split", "0");
    result += write_to_file(GPU_SYSFS_PATH + "/force_bus_on", "1");
    result += write_to_file(GPU_SYSFS_PATH + "/force_clk_on", "1");
    result += write_to_file(GPU_SYSFS_PATH + "/force_rail_on", "1");
    result += write_to_file(GPU_SYSFS_PATH + "/idle_time", "10000000");
    result += write_to_file(GPU_SYSFS_PATH + "/devfreq/governor", "performance");
    result += write_to_file(GPU_GPUBW_PATH + "/governor", "performance");
#endif
    return result < 0 ? -1 : 0; 
}

int GPUExecution::set_gpu_to_max(void) {
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


void GPUExecution::perform_microbenchmark(void) {
    ///!!!!!!!!!!!!!!!TEMPORARY?????????????????????????????????
    #ifdef F8131
        coreBenchResults[0] = Performance(15.4, 73.17);
        coreBenchResults[1] = Performance(14.81, 70.34);
        coreBenchResults[2] = Performance(14.52, 68.97);
        coreBenchResults[3] = Performance(12.99, 61.68);
        coreBenchResults[4] = Performance(12.31, 58.49);
        coreBenchResults[5] = Performance(9.47, 44.98);
        coreBenchResults[6] = Performance(6.15, 29.19);
        busBenchResults[0] = Performance(15.4, 73.17);
        busBenchResults[1] = Performance(14.4, 68.39);
        busBenchResults[2] = Performance(13.31, 63.22);
        busBenchResults[3] = Performance(11.68, 55.48);
        busBenchResults[4] = Performance(8.98, 42.66);
        busBenchResults[5] = Performance(8.12, 38.57);
        busBenchResults[6] = Performance(6.81, 32.37);
        busBenchResults[7] = Performance(5.29, 25.13);
        set_gpu_to_level(6,6);
    #endif
}



void GPUExecution::fine_tune(void) {
    RunModelSample();
    while(executionTime > latencyTarget) {
        // scale up the frequency setting that is causing the bottleneck
        if (currentCoreFreqIndex == 0) {
            if (currentBusFreqIndex == 0) {
                std::cerr << "Latency Target " << latencyTarget << " unachievable, " <<
                            "execution time now:  " << executionTime << std::endl;
                std::cerr << "fine tuning failed" << std::endl; 
            }
            else if (currentBusFreqIndex > 0 && currentBusFreqIndex < numBusFreq) {
                set_gpu_to_level(currentCoreFreqIndex, currentBusFreqIndex - 1);
            }
            else {
                std::cerr << "FATAL ERROR: Wrong bus freq index while fine tuning!!!" << std::endl;
                exit(-1);
            }
        }
        else if(currentCoreFreqIndex > 0 && currentCoreFreqIndex < numCoreFreq) {
            if(currentBusFreqIndex == 0) {
                set_gpu_to_level(currentCoreFreqIndex  - 1, currentBusFreqIndex);
            }
            else if(currentBusFreqIndex > 0 && currentBusFreqIndex < numBusFreq) {
                if(coreBenchResults[currentCoreFreqIndex - 1].computation <= 
                    busBenchResults[currentBusFreqIndex - 1].computation) {
                    set_gpu_to_level(currentCoreFreqIndex  - 1, currentBusFreqIndex);
                }
                else {
                    set_gpu_to_level(currentCoreFreqIndex, currentBusFreqIndex - 1);
                }
            }
            else {
                std::cerr << "FATAL ERROR: Wrong bus freq index while fine tuning!!!" << std::endl;
                exit(-1);
            }
        }
        else {
            std::cerr << "FATAL ERROR: Wrong core freq index while fine tuning!!!" << std::endl;
            exit(-1);
        }

        // recompute executionTime
        RunModelSample();
    }
    std::cout<< "Fine tuning: core " << coreFreq[currentCoreFreqIndex] <<
                " bus " << busFreq[currentBusFreqIndex] << std::endl;
}





absl::Status GPUExecution::RunModelSample(void) {
    auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
    GraphFloat32 graph_cl;
    ops::builtin::BuiltinOpResolver op_resolver;
    RETURN_IF_ERROR(BuildFromFlatBuffer(*flatbuffer, op_resolver, &graph_cl));

    Environment env;
    RETURN_IF_ERROR(CreateEnvironment(&env));

    InferenceContext::CreateInferenceInfo create_info;
    create_info.precision = env.IsSupported(CalculationsPrecision::F16)
                                ? CalculationsPrecision::F16
                                : CalculationsPrecision::F32;
    create_info.storage_type = GetFastestStorageType(env.device());
    std::cout << "Precision: " << ToString(create_info.precision) << std::endl;
    std::cout << "Storage type: " << ToString(create_info.storage_type)
            << std::endl;
    InferenceContext context;
    RETURN_IF_ERROR(
        context.InitFromGraphWithTransforms(create_info, &graph_cl, &env));

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

    //const int kNumRuns = 1;
    //for (int i = 0; i < kNumRuns; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < num_runs_per_sec; ++k) {
        RETURN_IF_ERROR(context.AddToQueue(env.queue()));
    }
    RETURN_IF_ERROR(env.queue()->WaitForCompletion());
    const auto end = std::chrono::high_resolution_clock::now();
    const double total_time_ms = (end - start).count() * 1e-6f;
    double average_inference_time = total_time_ms / num_runs_per_sec;
    std::cout << "Total time - " << average_inference_time << "ms" << std::endl;
    //}
    executionTime = average_inference_time;
    if(currentCoreFreqIndex == 0 && currentBusFreqIndex == 0 && executionTime > latencyTarget) {
        double oldTarget = latencyTarget;
        while(executionTime > latencyTarget) {
            latencyTarget *= 2.0f;
        }
        std::cerr << "Latency Target " << oldTarget << " unachievable, " <<
                    "lowering the latency target to " << latencyTarget << std::endl;
    }
    return absl::OkStatus();
}

absl::Status GPUExecution::RunPeriodically(const std::string& model_name) {
        auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
    GraphFloat32 graph_cl;
    ops::builtin::BuiltinOpResolver op_resolver;
    RETURN_IF_ERROR(BuildFromFlatBuffer(*flatbuffer, op_resolver, &graph_cl));

    Environment env;
    RETURN_IF_ERROR(CreateEnvironment(&env));

    InferenceContext::CreateInferenceInfo create_info;
    create_info.precision = env.IsSupported(CalculationsPrecision::F16)
                                ? CalculationsPrecision::F16
                                : CalculationsPrecision::F32;
    create_info.storage_type = GetFastestStorageType(env.device());
    std::cout << "Precision: " << ToString(create_info.precision) << std::endl;
    std::cout << "Storage type: " << ToString(create_info.storage_type)
            << std::endl;
    InferenceContext context;
    RETURN_IF_ERROR(
        context.InitFromGraphWithTransforms(create_info, &graph_cl, &env));

    auto* queue = env.profiling_queue();
    ProfilingInfo profiling_info;
    RETURN_IF_ERROR(context.Profile(queue, &profiling_info));

    const int num_runs_per_sec = std::max(
        1, static_cast<int>(1000.0f / absl::ToDoubleMilliseconds(
                                        profiling_info.GetTotalTime())));

    const auto start = std::chrono::high_resolution_clock::now();
    
    std::thread([]() {
        while(true) {
            auto wake_time = std::chrono::high_resolution_clock::now() + 
                    std::chrono::miliseconds()
            RETURN_IF_ERROR(context.AddToQueue(env.queue()));
            std::this_thread::sleep_until(wake_time);
        }
    })

    std::thread vision_timer([](){
        std::this_thread::sleep_for(wake_time);
    })

    vision_timer.join();
    
    
    /*for (int k = 0; k < num_runs_per_sec; ++k) {
        RETURN_IF_ERROR(context.AddToQueue(env.queue()));
    }
    RETURN_IF_ERROR(env.queue()->WaitForCompletion());*/
    const auto end = std::chrono::high_resolution_clock::now();


    return absl::OkStatus();
}
}

} // namespace cl
} // namespace gpu
} // namespace tflite

int main(int argc, char** argv) {
    double target_latency_ms;
    const std::string model_name;

    if (argc <= 1) {
        std::cerr << "Expected model path as second argument.";
        return -1;
    }
    if (argc <= 2) {
        std::cout << "Target performance not specified, defaulting to 60 fps" << std::endl;
        target_latency_ms = 1000.0f / 60.0f;
    }
    else {
        target_latency_ms = 1000.0f / std::atof(argv[2]);
    }

    auto load_status = tflite::gpu::cl::LoadOpenCL();
    if (!load_status.ok()) {
        std::cerr << load_status.message();
        return -1;
    }

    tflite::gpu::cl::GPUExecution execution(argv[1], target_latency_ms);
    execution.set_gpu_performance_mode();
    execution.set_gpu_to_max();

    auto run_status = execution.RunModelSample();
    if (!run_status.ok()) {
        std::cerr << run_status.message();
        return -1;
    }

    // TODO
    // microbenchmarking
    execution.perform_microbenchmark();

    // should be automatically decided by microbenchmarking results
    execution.fine_tune();


    execution.RunPeriodically();
}
