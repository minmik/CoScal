# F8131 GPU Frequency Manual Setting Guide
This article guides you how to manually set GPU frequency for Xperia X Performance (F8131). The following contents are tested on Ubuntu 18.04 and Xperia X Perforamcne build number 41.3.A.2.149 (Korean version) and is not guaranteed to work on other environments.  

## Step 1. Root the device
Root guide is provided in xda-developers forum by [korom42](https://forum.xda-developers.com/member.php?u=5033594).  
The guide and files needed for rooting can be found in this article: [https://forum.xda-developers.com/xperia-x-performance/how-to/guide-stock-kernel-root-twrp-drm-fix-41-t3711837](https://forum.xda-developers.com/xperia-x-performance/how-to/guide-stock-kernel-root-twrp-drm-fix-41-t3711837)  
The above guide is written for Windows users, so for Ubuntu, a little modification is needed.  
1. Unlock Bootloader  
Sony provides a guide for doing that. Just follow these two articles:  
[https://developer.sony.com/develop/open-devices/get-started/unlock-bootloader/](https://developer.sony.com/develop/open-devices/get-started/unlock-bootloader/)  
[https://developer.sony.com/develop/open-devices/get-started/unlock-bootloader/how-to-unlock-bootloader/](https://developer.sony.com/develop/open-devices/get-started/unlock-bootloader/how-to-unlock-bootloader/)  

2. Use Flashtool to Flash Oreo TFT  
Official website provides latest versions, but they might have some issues related to glibc. So, I found older releases that worked well in my case: 9.25.0 version.  
Download here: [https://androidmtk.com/download-sony-mobile-flasher](https://androidmtk.com/download-sony-mobile-flasher)  
How to download guide can be found here: [http://www.flashtool.net/lininstall.php](http://www.flashtool.net/lininstall.php)  
Some additional work must be done to use XperiFirm inside Flashtool to find adequate firmware for the device: [https://xperifirm.com/tutorial/install-xperifirm-linux/](https://xperifirm.com/tutorial/install-xperifirm-linux/)  
After preparing TFT file via XperiFirm, now use Flashtool to flash the device. Full wipe is recommened exclude nothing except TA files if there's any.  

3. When done flashing unplug the device and enter fastboot mode (hold vol up + plug usb) replug the deivce and flash TWRP:  
``` console
$ fastboot flash recovery twrp-3.2.1-0-dora.img
```
then flash kernel:  
``` console
$ fastboot flash boot boot.img
```

4. Unplug and enter TWRP (Hold vol down + power button)  
5. In TWRP flash drmfix.zip, then Magisk v16.3.zip  
6. Reboot the device: DONE!  

## Step 2. Workaround to modify GPU frequency
This is based on this article" https://programmersought.com/article/31261097363/.  
Refer above article for more information.  
The sysfs folder for the gpu (Adreno 530) can be found in  /sys/class/kgsl/kgsl-3d0. Available frequency levels and governors are:
``` console
F8131:/sys/class/kgsl/kgsl-3d0 # cat gpu_available_frequencies
624000000 560000000 510000000 401800000 315000000 214000000 133000000
F8131:/sys/class/kgsl/kgsl-3d0 # cat ./devfreq/available_governors
spdm_bw_hyp
cache_hwmon
mem_latency
bw_hwmon
msm-vidc-vmem+
msm-vidc-vmem
msm-vidc-ddr
bw_vbif
gpubw_mon
msm-adreno-tz
cpufreq
userspace
powersave
performance
simple_ondemand
```
The default governor is **msm-adreno-tz** which is basically more performance-oriented version of ondemand governor.

``` console
# su
# echo 1 >/sys/class/kgsl/kgsl-3d0/force_clk_on
# echo 10000000 >/sys/class/kgsl/kgsl-3d0/idle_timer
# echo performance >/sys/class/kgsl/kgsl-3d0/devfreq/governor
# echo [**want_clock**] > /sys/class/kgsl/kgsl-3d0/max_gpuclk
# echo [**want_clock**] > /sys/class/kgsl/kgsl-3d0/gpuclk
```
This will allow the su user to change the frequency. **force_clk_on** will set some flags. We disable GPU going into sleep by setting the **idle_timer** very high and use **performance** governor to set the frequency to the maximum specified by **max_gpuclk**. The reason why **userspace** governor is not used is that, even if **userspace** governor sets the frequency level at some point, the frequency level will change by some other features in the Adreno GPU while running some GPU kernels. Linux Kernel modification is required to maintain a fixed frequency by the **userspace** governor, so instead, we use this workaround.
