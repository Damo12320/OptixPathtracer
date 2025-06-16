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

    __device__ static bool IsEffectifvelySmooth(float r) {
        return r < 0.0001;
    }

    __device__ bool IsEffectifvelySmooth() {
        return IsEffectifvelySmooth(this->roughness);
    }
};