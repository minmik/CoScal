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

#ifndef DVFS_H
#define DVFS_H

#define F8131 1 // for test use

#define EXIT_IF_ERROR(s) {auto c=(s);if(!c.ok()){std::cerr << c.message();exit(-1);}}

#include <fstream>
#include <cstdlib>

#include <thread>
//#include <functional>


#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <iostream>
#include <string>
#include <time.h>

#include "absl/time/time.h"
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/testing/tflite_model_reader.h"
#include "tensorflow/lite/kernels/register.h"

namespace tflite {
namespace gpu {
namespace cl {

// Sony Xperia X Performance
// Snapdragon 820, Adreno 530 GPU
#ifdef F8131
const std::string CPU_SYSFS_PATH = "/sys/devices/system/cpu";
const std::string GPU_SYSFS_PATH = "/sys/class/kgsl/kgsl-3d0"; //< path for core sysfs
const std::string GPU_GPUBW_PATH = "/sys/class/devfreq/soc:qcom,gpubw"; //< path for bus devfreq
// available GPU core frequencies can be found at "GPU_SYSFS_PATH/available_frequencies"
// available GPU bus frequencies can be found at "GPU_GPUBW_PATH/available_frequencies"
// Although we can directly extract the values from the path
// We just manually define it here for convenience since they never change
const int numCoreFreq = 7;
const int numBusFreq = 12;

// available core frequencies in descending order
const std::string coreFreq [7] = {"624000000",
                                "560000000",
                                "510000000",
                                "401800000",
                                "315000000",
                                "214000000",
                                "133000000"};
// available bus frequencies in descending order
const std::string busFreq [12] = {"13763",
                                "11863",
                                "9887",
                                "7759",
                                "5859",
                                "5195",
                                "4173",
                                "3143",
                                "2288",
                                "1525",
                                "1144",
                                "762"};
#endif // F8131_DEFINED

// Pixel 4
// Snapdragon 855, Adreno 640v2 GPU
#ifdef FLAME
const std::string CPU_SYSFS_PATH = "/sys/devices/system/cpu";
const std::string GPU_SYSFS_PATH = "/sys/class/kgsl/kgsl-3d0"; //< path for core sysfs
const std::string GPU_GPUBW_PATH = "/sys/class/devfreq/soc:qcom,gpubw"; //< path for bus devfreq
// available GPU core frequencies can be found at "GPU_SYSFS_PATH/available_frequencies"
// available GPU bus frequencies can be found at "GPU_GPUBW_PATH/available_frequencies"
// Although we can directly extract the values from the path
// We just manually define it here for convenience since they never change
const int numCoreFreq = 5;
const int numBusFreq = 11;

// available core frequencies in descending order
const std::string coreFreq [5] = {"585000000",
                                "499200000",
                                "427000000",
                                "345000000",
                                "257000000"};
// available bus frequencies in descending order
const std::string busFreq [11] = {"7980",
                                "6881",
                                "5931",
                                "5161",
                                "3879",
                                "2929",
                                "2597",
                                "2086",
                                "1720",
                                "1144",
                                "762"};
#endif // FLAME_DEFINED

/**
 * @brief  helper function for writing into a file
 *
 * @param path  absolute file path to write to
 * @param value  value to write to the file
 *
 * @return 0  successful
 * @return -1  some error happened during the process (likely to be a permission error) 
 */
int write_to_file(const std::string path, const std::string value);

// Run tflite model, copyright: Tensorflow Authors
// Original Code: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/cl/testing/performance_profiling.cc
// modified by Jongmin Kim
/**
 * @brief  Run the Model on GPU on a specific interval using Tensorflow APIs simulating continuous vision workload
 * @param model_name  name of the CNN model
 * @param period  period of CNN inference in miliseconds
 */
void RunPeriodically(const std::string model_name, double period);


class GPUExecution {
public:
    /**
     * @brief  measured performance in microbenchmark
     */

    /* MAYBE will be used in mixbench
    struct Performance {
        float computation; //< computation perforamnce in GFLOPS
        float bandwidth;   //< bandwitdth in GB/sec
    };
    */

    GPUExecution() = delete;
    GPUExecution(const std::string _model_name, double target);
    ~GPUExecution() {
        delete[] coreBenchResults_ms;
        delete[] busBenchResults_ms;
    };
    
    // forbid copy or move
    GPUExecution(const GPUExecution &) = delete;
    GPUExecution & operator=(const GPUExecution &) = delete;
    
    
    /**
     * @brief  set GPU to performance mode as specified by vendors
     *
     * @return 0  successful 
     * @return -1  some error happened during the process (likely to be a permission error) 
     */
    int set_gpu_performance_mode(void);

    /**
     * @brief  set GPU frequencies to max
     *
     * @return 0  successful 
     * @return -1  some error happened during the process (likely to be a permission error) 
     */
    inline int set_gpu_to_max(void);

    /**
     * @brief  set GPU frequencies to certain level
     * @param core_index  target frequency index to the coreFreq array
     * @param bus_index  target frequency index to the busFreq array
     *
     * @return 0  successful 
     * @return -1  some error happened during the process (likely to be a permission error) 
     */
    int set_gpu_to_level(int core_index, int bus_index);

    /**
     * @brief  run the model for some frequency settings for benchmarking
     */
    void perform_benchmark(void);

    /**
     * @brief  set the dvfs frequencied according to the benchmark result
     */
    void set_dvfs_using_bench(void);

    /**
     * @brief  fine tune the GPU frequency.
     */
    void fine_tune(void);

    // Run tflite model, copyright: Tensorflow Authors
    // Original Code: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/cl/testing/performance_profiling.cc
    // modified by Jongmin Kim
    /**
     * @brief  Run the Model on GPU using Tensorflow APIs
     *
     * @return absl::OkStatus()  if successful, returns error otherwise
     */
    absl::Status RunModelSample(void);

private:
    std::string model_name;       //< model file name ***.tflite
    int currentCoreFreqIndex;           //< current index for coreFreq
    int currentBusFreqIndex;            //< current index for busFreq
    double executionTime;               //< execution latency for current settings in ms
    double latencyTarget;               //< latency target for GPU execution in ms

    // Tensorflow environment related variables
    std::unique_ptr<FlatBufferModel> flatbuffer; //< Flat BUffer Model
    GraphFloat32 graph_cl;
    ops::builtin::BuiltinOpResolver op_resolver;
    Environment env;
    InferenceContext context;
    InferenceContext::CreateInferenceInfo create_info;


    /* MAYBE will be used in mixbench
    double opIntensity;                 //< Operational intensity for microbenchmarking
                                        //  Operations/byte
    Performance * coreBenchResults;     //< measured performance for core DVFS microbenchmarking
    Performance * busBenchResults;      //< measured performance for bus DVFS microbenchmarking
    */
    double* coreBenchResults_ms;        //< measured latency for core DVFS benchmark
    double* busBenchResults_ms;         //< measrued latency for bus DVFS benchmark
};

} //namespace cl
} //namespace gpu
} //namespace tflite

#endif // DVFS_H_INCLUDED
