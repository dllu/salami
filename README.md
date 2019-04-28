SALAMI-SLAM

===

Inspired by IMLS-SLAM according to the description in the paper:

* Deschaud, J. E. (2018). IMLS-SLAM: scan-to-model matching based on 3D data. ICRA 2018. [arxiv:1802.0633](https://arxiv.org/pdf/1802.08633.pdf)

# Building

Before building, you need to install:

* [Eigen](https://eigen.tuxfamily.org/)
* [CMake](https://cmake.org/)

Then,

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE="Release"
make
```

To run on dataset 0, `mkdir -p poses && ./salami /path/to/kitti/dataset/sequences 0`
