#pragma once
#include <vector>

#include "../../3rdParty/glm/glm.hpp"
#include "../OpenGL/GLTexture2D.h"

namespace WriteImage {
	bool WriteBMP(std::vector<glm::vec3>& pixels, glm::ivec2 size, const char* path);
	void WriteEXR(std::vector<glm::vec3>& pixels, glm::ivec2 size, const char* path);



	void WriteTextureToBMP(GLTexture2D* texture, const char* path);
	void WriteTextureToEXR(GLTexture2D* texture, const char* path);
}