#pragma once
#include <vector>

#include "../../3rdParty/glm/glm.hpp"

namespace WriteImage {
	bool WriteBMP(std::vector<uint64_t>* pixels, glm::ivec2 size, const char* path);
}