# storage-ring-gravitational-wave-observatory
This code simulates the expected response of the SRGO detector to the dominant harmonic of millihertz gravitational waves from the inspiral phase of a non-spinning binary system merger, accounting for the effect of Earth's rotation. It also produces the GW source localization sky-maps by fitting the response model to artificial noisy data points using an MCMC (Markov Chain Monte Carlo) method, the way LIGO does. There are also functionalities to obtain and plot the SRGO sensitivity curve, observational range, etc. Authors: Suvrat Rao, Hamburg Observatory, University of Hamburg; Julia Baumgarten, Physics department, Jacobs University Bremen. Reference: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.122006

STEP 1: Rename the "theanorc.txt" file to ".theanorc.txt" (notice the '.' at the start) and place this file in your "C:\Users\username" folder. Also, rename the paths inside this file to the corresponding paths of your computer. Some paths will only exist after you complete the installation procedure following the guide below. If you do not wish to set up GPU assist, then skip those steps from the guide and delete the appropriate lines from .theanorc.txt. If you are confused about this, simply delete .theanorc.txt, and the code should still work (without GPU assist). 

STEP 2: Create a new folder named "saved_plots" (results will automatically be saved to this folder) and place it in the same folder as "srgo.py" along with "earth.png", which is required by the code to create the main plots. The other images are also used in making plots, but they are re-generated and saved by the code with each run.

STEP 3: ORDERWISE, PROPER MODULE INSTALLATION, ERROR DEBUGGING & OPTIMIZATION INSTRUCTIONS <br />

*FOR WINDOWS (with GPU assist)*: <br />
Start with a clean installation, then just follow this list mechanically and you should be good. Also, skip making separate environments and just install everything in the anaconda base environment. <br />
https://gist.github.com/ElefHead/93becdc9e99f2a9e4d2525a59f64b574 <br />
(^ skip any module installs mentioned in this & copy all the cuDNN folder contents to the CUDA folder) <br />
https://stackoverflow.com/questions/57701571/what-is-the-right-way-to-update-anaconda-and-conda-base-environments <br />
(^ conda update conda ; conda update anaconda ; conda install spyder=5.0.5 ; conda update --all) <br />
https://github.com/pymc-devs/pymc3/wiki/Installation-Guide-(Windows) <br />
(^ in line 1, skip "python=3.8" and scipy ; but add pygpu) <br />
https://www.programmersought.com/article/18283822507/ <br />
https://stackoverflow.com/questions/28011551/how-configure-theano-on-windows <br />
https://theano-pymc.readthedocs.io/en/latest/library/config.html <br />
https://github.com/conda-forge/magma-feedstock <br />
https://theano-pymc.readthedocs.io/en/latest/tutorial/using_gpu.html <br />
https://theano-pymc.readthedocs.io/en/latest/troubleshooting.html <br />
https://docs.huihoo.com/theano/0.8/faq.html <br />
https://stackoverflow.com/questions/50949520/cudnn-path-not-resolving <br />
(^ library_path = ...\lib64\x64) <br />
https://github.com/monero-project/monero/issues/3521 <br />
https://docs.microsoft.com/en-us/cpp/assembler/masm/masm-for-x64-ml64-exe?view=msvc-160 <br />
(^ already present in MS Visual Studio and added to path in step 1, so directly call editbin.exe in cmd) <br />
https://www.windowscentral.com/how-change-virtual-memory-size-windows-10?amp <br />
In case of any __init__.py errors, open those files and hotfix the respective lines. 

*FOR LINUX (without GPU assist)*: <br />
The 'seaborn' and 'OpenCV' python modules are needed. Apart from that, simply do "pip install pymc3" and you're good to go!


NOTE1: Be careful if you wish to time the code, as it may hang the MCMC subroutine due to unknown reasons!

NOTE2: If your plots or saved figures look misaligned, you must change your matplotlib backend to automatic (https://stackoverflow.com/questions/23585126/how-do-i-get-interactive-plots-again-in-spyder-ipython-matplotlib)

NOTE3: The MCMC results are considered accurate if the true parameter values lie within the 90% HPDI (highest posterior density interval) region, 90% of the time. Therefore, you must run the code several times to achieve accurate results.
