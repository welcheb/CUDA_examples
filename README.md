### Step 1: Flash a TK1-specific Ubuntu 14.04 release
1. Followed instructions available at [https://cyclicredundancy.wordpress.com/2014/05/10/](https://cyclicredundancy.wordpress.com/2014/05/10/)
2. True requirements are:
	* separate linux host with ample free disk space (~16GB) (failed using a raspberry pi as the host because of not enough disk space)
	* USB cable included with the TK1
3. Need to download two files with very similar names. Files available at this top-level page:
[https://developer.nvidia.com/embedded/linux-tegra-archive](https://developer.nvidia.com/embedded/linux-tegra-archive) Be careful to avoid stuff for TX1 (the other kit). For example, on the "Jetson TK1 R21.4 - July 2015" page, you will want the download under "Driver Packages" [http://developer.download.nvidia.com/embedded/L4T/r21_Release_v4.0/Tegra124_Linux_R21.4.0_armhf.tbz2](http://developer.download.nvidia.com/embedded/L4T/r21_Release_v4.0/Tegra124_Linux_R21.4.0_armhf.tbz2) and the download under "Sample File System" [http://developer.download.nvidia.com/embedded/L4T/r21_Release_v4.0/Tegra_Linux_Sample-Root-Filesystem_R21.4.0_armhf.tbz2](http://developer.download.nvidia.com/embedded/L4T/r21_Release_v4.0/Tegra_Linux_Sample-Root-Filesystem_R21.4.0_armhf.tbz2).
4. Once you have the above two files, you can perform all the steps described at https://cyclicredundancy.wordpress.com/2014/05/10/ I recommend using the updated flash command that uses the full space on the eMMC, i.e.,

~~~~
sudo ./flash.sh -S 14580MiB jetson-tk1 mmcblk0p1
~~~~

* Don't forget to put the development board into force recovery mode by holding down the “RECOVERY” button and pressing the “RESET” button once.
* The untaring, binary building and flashing take a while. In total, it will take at least an hour. When it is complete, the board will reboot and show a login window for the Ubuntu desktop (default username: ubuntu default password: ubuntu). Then you can move on to installing CUDA. You can also perform some of the steps suggested at the bottom of the wordpress page to get common utilities and force the system and hardware clocks to be set correctly.

### Step 2: Install CUDA
1. The true requirements are:
	* Already flashed TK1 as created in Step 1 above (the CUDA install is performed when logged into the TK1)
2. Follow the steps described here: [http://elinux.org/Jetson/Installing_CUDA](http://elinux.org/Jetson/Installing_CUDA)
3. Be sure to get the L4T debian package created for the TK1 kit. You can download it back at the same page where you downloaded the "Driver Packages" and "Sample File System" under the heading "CUDA 6.5 Toolkit for L4T Rel 21.4". I downloaded the specific file [http://developer.download.nvidia.com/embedded/L4T/r21_Release_v3.0/cuda-repo-l4t-r21.3-6-5-prod_6.5-42_armhf.deb](http://developer.download.nvidia.com/embedded/L4T/r21_Release_v3.0/cuda-repo-l4t-r21.3-6-5-prod_6.5-42_armhf.deb)
4. Once I completed the CUDA install instructions, I could use nvcc to compile.



#### *Note* ####
The TK1 user guide also refers to installing/flashing something called JetPack that seems to already include a bunch of CUDA development tools. I never tried it. It seems to specifically need an Ubuntu 14.04 host (which I did not have handy).
