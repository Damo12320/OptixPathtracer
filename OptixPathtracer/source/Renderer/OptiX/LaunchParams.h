#pragma once

#include "../../3rdParty/OptixSample/optix7.h"
#include "../../3rdParty/glm/glm.hpp"

enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

struct LaunchParams
{
    struct {
        uint32_t* colorBuffer;
        glm::ivec2     size;
    } frame;

    struct {
        glm::vec3 position;
        glm::vec3 direction;
        glm::vec3 horizontal;
        glm::vec3 vertical;
    } camera;

    OptixTraversableHandle traversable;
};