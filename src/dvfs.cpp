#include "dvfs.h"

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
#ifdef F8131
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
    int result = -1;
#ifdef F8131
    result = 0;
    result += write_to_file(GPU_SYSFS_PATH + "/max_gpuclk", coreFreq[0]);
    result += write_to_file(GPU_GPUBW_PATH + "/max_freq", busFreq[0]);
    result += write_to_file(GPU_GPUBW_PATH + "/min_freq", busFreq[0]);
#endif
    return result < 0 ? -1 : 0;
}

int main() {
    GPUExecution execution;
    execution.set_gpu_performance_mode();
    execution.set_gpu_to_max();
}
