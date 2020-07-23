# dvfs_project
Use DVFS to optimize energy for CNN applications

## Problem Formulation
A heterogeneous mobile system can be abstracted into just a pool of available processors.
Take Pixel 4 for a running example.

|   |   |
|---|---|
|CPU big     | 1  ×  2.84  GHz Kryo 485 Gold Prime |
|CPU medium  | 3  ×  2.42  GHz Kryo 485 Gold |
|CPU little  | 4  ×  1.78  GHz Kryo 485 Silver|
|GPU         | Adreno 640 |
|NPU         | Pixel Neural Core (TPU) |

We can make an abstraction of this system like this:  

![problem0](./img/readme_image0.svg)

and each type of processor shows some properties related to running CNN. We'll define symbols as such:  

*DVFS_available*: whether DVFS can be used for this processor. Most NPUs do not support DVFS.  
*V, f* : Voltage and frequency that can be selected by DVFS system.  
*n* : number of processors being used by the application for a certain type [ex] *n_k* for CPU *C_k*  
*Th* : Throughput achieved by a processor running at certain V & f.  [ex] *Th ( C2, V, f )*  
*E* : Energy required to process one image by a processor running at certain V & f. [ex] *E_k( V, f )* for CPU *C_k*  

and the application has these requirements:  

*Th_tar* : Target throughput  
*E_max* : Energy budget  

NPU and GPU are prioritized under the assumption that they are superior in energy efficiency than CPU. Actual implementation is done with Android NNAPI. However, NPU and GPU are not always available. So, CPU usage should be considered for such cases.  
The throughput attainable from using NPU and GPU will be defined as *Th_NNAPI*  
then the problem can be defined as:  

![problem1](./img/readme_image1.svg)

Our DVFS plan uses an approach from [1], where critical speed, which is defined as the least energy consuming frequency of processor operation for a given memory access rate (MAR). The difference is that we use *Memory Access / FLOPS* statically decided by the model instead of MAR, getting rid of the use of online PMU counter usage. Also, in stead of using only the critical speed (which actually uses the combination of two most close frequencies possible when given a target critical speed for an application's MAR), we allow some higher frequencies which shows better performance at increased energy cost. This modification allows adaptation for performance target more flexible.

Overall research is expected to be done on this order:  
1. Create a MAR-CSE like regression model by extensive energy profiling on each type of core.
2. Create an Android framework for running CNN models on various processors when given a throughput target. (There's already a well-known example for running TensorFlow Lite models on CPU/GPU/NPU by using NNAPI)
3. Use critical speed when the model is run on CPUs.
4. Combine various types of cores for deciding where to run the CNN model.
5. Use some extra higher frequencies other than the critical speed when the throughput difference is small.
6. Test for various circumstances, models, devices, number of extra frequencies allowed.  

We are going to use many materials from TensorFlow Lite from benchmarking, optimization, implementation for easy use and porting.



## References

[1] Liang, Wen-Yew, and Po-Ting Lai. "Design and Implementation of a Critical Speed-based DVFS Mechanism for the Android Operating System." 2010 5th International Conference on Embedded and Multimedia Computing. IEEE, 2010.

## Writer
**Jongmin Kim**
Seoul National University, Dept. of Electrical and Computer Engineering
Research Intern at [Networked Computing Lab.](https://nxc.snu.ac.kr/)
GitHub: [github.com/minmik](https://github.com/minmik)
Email: kimcm3310@gmail.com
