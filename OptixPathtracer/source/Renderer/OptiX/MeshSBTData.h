#pragma once

#include "../../3rdParty/OptixSample/optix7.h"
#include "../../3rdParty/glm/glm.hpp"

struct MeshSBTData {
    glm::vec3  color;
    glm::vec3* vertex;
    glm::vec3* normal;
    glm::vec2* texcoord;
    glm::ivec3* index;
    bool                hasTexture;
    cudaTextureObject_t texture;
};