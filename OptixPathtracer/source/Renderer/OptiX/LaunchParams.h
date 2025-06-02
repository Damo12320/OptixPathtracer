#pragma once

#include "../../3rdParty/OptixSample/optix7.h"
#include "../../3rdParty/glm/glm.hpp"

enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

struct LaunchParams
{
    struct {
        uint32_t* colorBuffer;
        glm::ivec2     size;
        unsigned int id = 0;
    } frame;

    struct {
        glm::vec3 position;
        glm::mat4x4 inverseViewMatrix;
        glm::mat4x4 inverseProjectionMatrix;
    } camera;

    OptixTraversableHandle traversable;
};