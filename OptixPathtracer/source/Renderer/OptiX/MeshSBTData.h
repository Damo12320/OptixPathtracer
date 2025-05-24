#pragma once

#include "../../3rdParty/OptixSample/optix7.h"
#include "../../3rdParty/glm/glm.hpp"

struct MeshSBTData {
    glm::vec3  albedoColor;

    glm::vec3* vertex;
    glm::vec3* normal;
    glm::vec2* texcoord;
    glm::ivec3* index;

    glm::mat4x4 modelMatrix;

    float roughness;
    float metallic;

    bool hasAlbedoTexture;
    bool hasNormalTexture;
    bool hasMetalRoughTexture;

    cudaTextureObject_t albedoTexture;
    cudaTextureObject_t normalTexture;
    cudaTextureObject_t metalRoughTexture;
};