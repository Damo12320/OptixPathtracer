#pragma once
#include <vector>
#include <glad/glad.h>
#include "3rdParty/glm/glm.hpp"

class OptixViewTexture {
public:
	GLuint handle;
public:
	OptixViewTexture();
	OptixViewTexture(uint32_t pixels[], glm::ivec2 size);

	void SetData(uint32_t pixels[], glm::ivec2 size);

	void BindTexture();
};