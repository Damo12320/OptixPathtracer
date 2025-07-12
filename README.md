# Physically based Pathtracer
This is a physically based Pathtracer using [NVIDIA Optix](https://developer.nvidia.com/rtx/ray-tracing/optix). 
It uses an OpenGL Context for image preview and processing, but the rendered image is calculated using OptiX.


This implementation is based on the code and concepts presented by Matt Phar, Wenzel Jakob, and Greg Humphreys in the Physically Based Rendering Book.

PBRT Book: https://www.pbr-book.org/

PBRT Repository: https://github.com/mmp/pbrt-v4



## Features
The Focus of this implementation lies on the BRDFs. This is reflected in the featureset:
- Conductor BRDF
- Diffuse BRDF
- Dielectric BSDF
- Layered BRDF
  - Diffuse base with a dielectric coating
- glTF loading
- OpenGL Window Preview





## Analysis

This implementation is compared to the PBRT implementation to analyse the validity of it. The Image Tools from PBRT where used to determine the Metrics values. 
It shows, that this implementation still has some differences to the PBRT implementation, but it seems to be close.

Bigger differences:
- The conductor BRDF from PBRT has a white effect around the mirror spheres, while this implementation doesn't show this.
- The layered BRDF from this implementation is a bit brighter than the PBRT version.

Lower values are better.
| Scene                            | Mean Squared Error (MSE)       | [FLIP](https://research.nvidia.com/publication/flip) |
| -------------------------------- | ------------------------------ | ---------------------------------------------------- |
| Conductor                        | 2.57E-07                       |	0.0017468039                                         |
| Diffuse                          | 4.59E-06                       |	0.006595257                                          |
| Dielectric (dark)                | 1.24E-07                       |	0.00084382464                                        |
| Dielectric (bright)              | 4.98E-05                       |	0.007553334                                          |
| Layered (Diffuse + Dielectric)   | 8.84E-07                       | 0.0032398894                                         |

## Images

|Scene                            | This implementation                                         | PBRT                                                           |
|-------------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------- |
| Conductor                       | ![Conductor Mine](/Images/PNGs/conductor.png)               | ![Conductor PBRT](/Images/PNGs/conductorPBRT.png)              |
| Diffuse                         | ![Diffuse Mine](/Images/PNGs/diffuse.png)                   | ![Diffuse PBRT](/Images/PNGs/diffusePBRT.png)                  |
| Dielectric (dark)               | ![Dielectric (dark) Mine](/Images/PNGs/dielectric.png)      | ![Dielectric (dark) PBRT](/Images/PNGs/dielectricPBRT.png)     |
| Dielectric (bright)             | ![Dielectric (bright) Mine](/Images/PNGs/dielectric20.png)  | ![Dielectric (bright) PBRT](/Images/PNGs/dielectric20PBRT.png) |
| Layered (Diffuse + Dielectric)  | ![Layered Mine](/Images/PNGs/layered.png)                   | ![Layered PBRT](/Images/PNGs/layeredPBRT.png)                  |



















