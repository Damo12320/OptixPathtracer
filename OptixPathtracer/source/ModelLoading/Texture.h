#pragma once
#include <cstdint>
#include "../3rdParty/glm/glm.hpp"

struct Texture {
    ~Texture()
    {
        if (pixel) delete[] pixel;
    }

    uint32_t* pixel{ nullptr };
    glm::ivec2 resolution{ -1 };
};