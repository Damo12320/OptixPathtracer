#pragma once

#include <glad/glad.h>
#include "../../3rdParty/stb_image.h"
#include "../../3rdParty/glm/glm.hpp"
#include <iostream>
#include <cmath>

class GLTexture2D {
private:
	unsigned int ID;

	int width;
	int height;
public://Getters and Setters
	unsigned int GetID() { return this->ID; }
	int GetWidth() { return this->width; }
	int GetHeight() { return this->height; }
public:
	GLTexture2D(const char* texturePath);
	GLTexture2D(glm::ivec2 size, int sizedInternalFormat = GL_RGBA8, int mipMapLevels = 1);

	~GLTexture2D();

	void BindToUnit(GLuint unit);
	void Bind();

	void SetData(uint32_t pixels[], glm::ivec2 size);
};