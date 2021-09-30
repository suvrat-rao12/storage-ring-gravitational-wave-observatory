# storage-ring-gravitational-wave-observatory
Simulating the detection of millihertz (mHz) gravitational waves (GWs) from astrophysical sources by a Storage Ring Gravitational-wave Observatory (SRGO).

NOTE1: Rename the "theanorc.txt" file to ".theanorc.txt" (notice the '.' at the start) and place this file in your "C:\Users\username" folder. Also rename the paths inside this file to the corresponding paths of your computer. Some paths will only exist after you complete the installation procedure following the guide below. 

NOTE2: Place "earth.png" in the same folder as "srgo.py". It is required by the code to create the main plots. The other images are also used in making plots, but they are re-generated and saved by the code with each run.

NOTE3: Be careful if you wish to time the code, as it may hang the MCMC subroutine due to unknown reasons!

ORDERWISE, PROPER MODULE INSTALLATION, ERROR DEBUGGING & OPTIMIZATION INSTRUCTIONS *FOR WINDOWS*:
(start with a clean installation, then just follow this list mechanically and you should be good...)
(also, skip making separate environments and just install everything in the anaconda base environment)
https://gist.github.com/ElefHead/93becdc9e99f2a9e4d2525a59f64b574
(^ skip any module installs mentioned in this & copy all the cuDNN folder contents to the CUDA folder)
https://stackoverflow.com/questions/57701571/what-is-the-right-way-to-update-anaconda-and-conda-base-environments
(^ conda update conda ; conda update anaconda ; conda install spyder=5.0.5 ; conda update --all)
https://github.com/pymc-devs/pymc3/wiki/Installation-Guide-(Windows)
(^ in line 1, skip "python=3.8" and scipy ; but add pygpu)
https://www.programmersought.com/article/18283822507/
https://stackoverflow.com/questions/28011551/how-configure-theano-on-windows
https://theano-pymc.readthedocs.io/en/latest/library/config.html
https://github.com/conda-forge/magma-feedstock
https://theano-pymc.readthedocs.io/en/latest/tutorial/using_gpu.html
https://theano-pymc.readthedocs.io/en/latest/troubleshooting.html
https://docs.huihoo.com/theano/0.8/faq.html
https://stackoverflow.com/questions/50949520/cudnn-path-not-resolving
(^ library_path = ...\lib64\x64)
https://github.com/monero-project/monero/issues/3521
https://docs.microsoft.com/en-us/cpp/assembler/masm/masm-for-x64-ml64-exe?view=msvc-160
(^ already present in MS Visual Studio and added to path in step 1, so directly call editbin.exe in cmd)
https://www.windowscentral.com/how-change-virtual-memory-size-windows-10?amp
https://github.com/arviz-devs/arviz/pull/1665
In case of any __init__.py errors, open those files and hotfix the respective lines.
