# Notes on enabling Tensorflow/CUDA/GPU support through WSL2

## Inside Ubuntu 22 on WSL2
 - Install CUDA 11.8, not 12.X or a higher version, which doesn't seem to work.
 - Follow https://www.tensorflow.org/install/pip#windows-wsl2
  - Don't use conda, but venc.
  - Store the `env_vars.sh` file locally and not within the dir they give you. You can do `source env_vars.sh` later.
 - Follow this for 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
   - Instead of install cuda at the end, install cuda-11-8
    - See https://askubuntu.com/questions/1394352/force-cuda-toolkit-version-11-6-in-ubuntu-18-04-latest-in-repos-is-9-1