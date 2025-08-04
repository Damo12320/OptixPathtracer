#pragma once

#include "../../3rdParty/glm/glm.hpp"

struct Surface {
    glm::mat3 ShadingToWorld, WorldToShading;

    glm::vec3 gNormal, sNormal;
    glm::ivec3 index;
    glm::vec3 vertices[3];
    glm::vec3 position;
    glm::vec2 texCoord;

    glm::vec3 outgoingRay;

    float metallic, roughness;

    glm::vec3 albedo;

    bool conductor;

    __device__ __host__ static bool IsEffectifvelySmooth(float alpha) {
        return alpha < 1e-3f;
    }

    __device__ __host__ static float GetAlpha(float roughness) {
        return Sqr(roughness);
        //return glm::sqrt(roughness);
        //return roughness;
    }

    __device__ __host__ bool IsEffectifvelySmooth() {
        return IsEffectifvelySmooth(this->GetAlpha());
    }

    __device__ __host__ float GetAlpha() {
        return GetAlpha(this->roughness);
    }
};