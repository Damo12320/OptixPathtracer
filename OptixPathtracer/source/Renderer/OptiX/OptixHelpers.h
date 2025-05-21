#pragma once

#include <cuda_runtime.h>
#include "../../3rdParty/glm/glm.hpp"

namespace OptixHelpers {
    static __device__ float3 Float3(glm::vec3 vector) {
        float3 vec{ vector.x, vector.y, vector.z };
        return vec;
    }

    static __device__ float4 Float4(glm::vec4 vector) {
        float4 vec{ vector.x, vector.y, vector.z, vector.w };
        return vec;
    }

    static __device__ glm::vec3 Vec3(float3 vector) {
        return glm::vec3(vector.x, vector.y, vector.z);
    }

    static __device__ glm::vec4 Vec4(float4 vector) {
        return glm::vec4(vector.x, vector.y, vector.z, vector.w);
    }

    static __device__ glm::vec3 Add(glm::vec3 vector1, glm::vec3 vector2) {
        return glm::vec3(
            vector1.x + vector2.x,
            vector1.y + vector2.y,
            vector1.z + vector2.z
        );
    }
}