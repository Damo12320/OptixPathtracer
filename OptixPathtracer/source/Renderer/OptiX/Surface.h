#pragma once

#include "../../3rdParty/glm/glm.hpp"

struct Surface {
    glm::mat3 ShadingToWorld, WorldToShading;

    glm::vec3 gNormal, sNormal, stangent, sbitangent;
    glm::ivec3 index;
    glm::vec3 vertices[3];
    glm::vec3 position;
    glm::vec2 texCoord;

    glm::vec3 incommingRay;

    float metallic, roughness;

    glm::vec3 albedo;

    __device__ bool IsEffectifvelySmooth() {
        return this->roughness < 0.0001;
    }
};