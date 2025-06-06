#pragma once

#include "../glmCUDA.h"

namespace PBRT {
    namespace SpherGeom {

        __device__ float CosTheta(glm::vec3 w) { return w.z; }
        __device__ float Cos2Theta(glm::vec3 w) { return w.z * w.z; }
        __device__ float AbsCosTheta(glm::vec3 w) { return std::abs(w.z); }

        __device__ float Sin2Theta(glm::vec3 w) { return glm::max(0.0f, 1.0f - Cos2Theta(w)); }
        __device__ float SinTheta(glm::vec3 w) { return sqrt(Sin2Theta(w)); }

        __device__ float TanTheta(glm::vec3 w) { return SinTheta(w) / CosTheta(w); }
        __device__ float Tan2Theta(glm::vec3 w) { return Sin2Theta(w) / Cos2Theta(w); }

        __device__ float CosPhi(glm::vec3 w) {
            float sinTheta = SinTheta(w);
            return (sinTheta == 0.0f) ? 1.0f : glm::clamp(w.x / sinTheta, -1.0f, 1.0f);
        }
        __device__ float SinPhi(glm::vec3 w) {
            float sinTheta = SinTheta(w);
            return (sinTheta == 0.0f) ? 0.0f : glm::clamp(w.y / sinTheta, -1.0f, 1.0f);
        }

        __device__ bool SameHemisphere(glm::vec3 w, glm::vec3 wp) {
            return w.z * wp.z > 0.0f;
        }
    }
}