# DenseMapPosterior.jl

This is a [Julia](https://www.julialang.org/) implementation of the DenseMapPosterior metric presented in our paper:
```
G. Zhang and Y. Chen, "A Metric For Evaluating 3D Reconstruction And Mapping Performance With No Ground Truthing," 2021 IEEE International Conference on Image Processing (ICIP), 2021, pp. 3178-3182, doi: 10.1109/ICIP42928.2021.9506329.
```
```
@InProceedings{Zhang2021,
  author    = {Guoxiang Zhang and YangQuan Chen},
  booktitle = {2021 {IEEE} International Conference on Image Processing ({ICIP})},
  title     = {{A Metric For Evaluating 3D Reconstruction And Mapping Performance With No Ground Truthing}},
  year      = {2021},
  month     = sep,
  publisher = {{IEEE}},
  doi       = {10.1109/icip42928.2021.9506329},
}
```
Please cite our paper if you use it in your work.
# Installation
This package is tested with Julia 1.6. To install, open julia's REPL and run the following code
(Don't worried if you never used Julia before. You will be amazed by how easy it is to get this package running.)

Note: This implementation requires an NVIDIA GPU for model fusion, virtual observations and fast calculation. It only runs on Linux x64 at the moment.

```julia
using Pkg

Pkg.add("CUDA")
Pkg.add("ModernGL")

Pkg.add(url="https://github.com/gzhang8/Pangolin_jll.jl.git")
Pkg.add(url="https://github.com/gzhang8/PangolinCApi_jll.jl.git")
Pkg.add(url="https://github.com/gzhang8/Pangolin.jl.git")

Pkg.add(url="https://github.com/gzhang8/ElasticFusionLib_jll.jl.git")
Pkg.add(url="https://github.com/gzhang8/ElasticFusion.jl.git")

Pkg.add(url="https://github.com/gzhang8/CameraModel.jl.git")

Pkg.add(url="https://github.com/gzhang8/GlHelper.jl.git")
Pkg.add(url="https://github.com/gzhang8/SurfelsLib_jll.jl.git")
Pkg.add(url="https://github.com/gzhang8/Surfels.jl.git")

# The following two pakcage will be cloned into "DMP" folder in your home folder
# Change the "DMP" to your desired path if you want
ENV["JULIA_PKG_DEVDIR"] = joinpath(homedir(), "DMP")

Pkg.develop(url="https://github.com/gzhang8/SLAMData.jl.git")

Pkg.develop(url="https://github.com/gzhang8/DenseMapPosterior.jl.git")
```

# Usage

To run the sample program, first download the data [here](https://drive.google.com/file/d/1_9S50Utbl6NftU8_I92J2sH5La-9NOEm/view?usp=sharing). Then unzip and place the `livingroom2.klg` to the `data/livingroom2` folder. Then  
```
julia apps/eval_aug_icl_livingroom2.jl
```