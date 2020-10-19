#ifndef DVFS_H
#define DVFS_H

#define F8131 1 // for test use

#include <iostream>
#include <fstream>
#include <string>

// Sony Xperia X Performance
// Snapdragon 820, Adreno 530 GPU
#ifdef F8131
const std::string GPU_SYSFS_PATH = "/sys/class/kgsl/kgsl-3d0"; //< path for core sysfs
const std::string GPU_GPUBW_PATH = "/sys/class/devfreq/soc:qcom,gpubw"; //< path for bus devfreq
// available GPU core frequencies can be found at "GPU_SYSFS_PATH/available_frequencies"
// available GPU bus frequencies can be found at "GPU_GPUBW_PATH/available_frequencies"
// Although we can directly extract the values from the path
// We just manually define it here for convenience since they never change
const int numCoreFreq = 7;
const int numBusFreq = 12;
const std::string coreFreq [7] = \ //< available core frequencies in descending order
{"624000000","560000000", "510000000", "401800000", "315000000", "214000000", "133000000"}
const std::string busFreq [12] = \ //< available bus frequencies in descending order
{"13763", "11863", "9887", "7759", "5859", "5195", "4173", "3143", "2288", "1525", "1144", "762"};
#endif // F8131_DEFINED

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



/**
 * @brief  microbenchmarking class
 */
class Microbenchmark {
public:
    /**
     * @brief  measured performance in microbenchmark
     */
    struct Performance {
        float computation; //< computation perforamnce in GFLOPS
        float bandwidth;   //< bandwitdth in GB/sec
    };

    Microbenchmark();
    ~Microbenchmark();

    // prevent copy or move
    Microbenchmark(const Microbenchmark &) = delete;
    Microbenchmark & operator=(const Microbenchmark &) = delete;

    int setTarget();
    
    int performBenchmark();

private:
    float opIntensity;              //< Operational intensity for microbenchmarking
                                    //  Operations/byte
    Performance * coreDVFSResults;  //< measured performance for core DVFS
    Performance * busDVFSResults;   //< measured performance for bus DVFS
};


class GPUExecution {
public:
    GPUExecution();
    ~GPUExecution();
    
    // prevent copy or move
    GPUExecution(const GPUExecution &) = delete;
    GPUExecution & operator=(const GPUExecution &) = delete;
    
    
    /**
     * @brief  set GPU to performance mode as specified by vendors
     *
     * @return 0  successful 
     * @return -1  some error happened during the process (likely to be a permission error) 
     */
    int set_gpu_performance_mode(void);


    int set_gpu_to_max(void);
private:
    std::string currentCoreFreq;
    std::string currentBusFreq;
};




#endif // DVFS_H_INCLUDED
