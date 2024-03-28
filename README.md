# NeRF-KPlanes-SemX

 An implementation and explanantion of NeRF and K-Planes in PyTorch. This repository contains the code for NeRF and K-Planes for my Semester X Neural Networks and Deep Learning Project. It is based on the implementation of Maxime Vandegar (Papers in 100 lines of code).

**Refer to full_pipe.ipynb for the complete pipeline and outputs of the project.**

**Report availabe in the report directory**

# ⚠️ README File WIP

## High Fidelity Outputs (Using NeRFStudio)

**Animations made using Blender**

| Guitar  | Shoe | Model Car|
| -------- | ------- | ------- |
| ![Preview](./img/Guitar.gif "Preview") | ![Preview](./img/Shoe.gif "Preview") |  ![Preview](./img/Car.gif "Preview") |

## Models Implemented

## Results

| NeRF | K-Planes |
| -------- | ------- |
| ![Result](./img/metric/nerf/nerf_0.png "Result") | ![Result](./img/metric/kplanes/kplanes_0.png "Result") |
| ![Result](./img/metric/nerf/nerf_1.png "Result") | ![Result](./img/metric/kplanes/kplanes_1.png "Result") |
| ![Result](./img/metric/nerf/nerf_2.png "Result") | ![Result](./img/metric/kplanes/kplanes_2.png "Result") |
| ![Result](./img/metric/nerf/nerf_3.png "Result") | ![Result](./img/metric/kplanes/kplanes_3.png "Result") |

### Averaged Metrics

| Metric | NeRF | K-Planes |
| -------- | ------- |  ------- |
| **MSE** | 29.92 | **26.63** |
| **PSNR** | **0.0010** | 0.0022 |
| **SSIM** | **0.99** | 0.99 |
| **Render Time** | 10.13s | **3.90s** |

## Repository Structure

### Directories

* report - Detailed report about project.
* img - Assets for readme.
* backbones - Scripts for NeRF and K-Planes.
* scraps - Notebook experimenting with Ray Tracing and Rendering.
* utils - Basic image utils.

### Scripts

* train.py - Training NeRF model.
* test.py - Novel View synthesis.
* evaluate.py - Metricced Novel View Synthesis.
* full_pipe.ipynb - Complete pipeline demo notebook.
