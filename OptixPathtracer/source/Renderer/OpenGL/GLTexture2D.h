#pragma once

#include <glad/glad.h>
#include "../../3rdParty/stb_image.h"
#include "../../3rdParty/glm/glm.hpp"
#include <iostream>
#include <cmath>
#include <vector>

class GLTexture2D {
private:
	unsigned int ID;

	int width;
	int height;
	int sizedInternalFormat;
	int mipMapLevels;
public://Getters and Setters
	unsigned int GetID() { return this->ID; }
	int GetWidth() { return this->width; }
	int GetHeight() { return this->height; }
	int GetSizedInternalFormat() { return this->sizedInternalFormat; }
	int GetMipMapLevels() { return this->mipMapLevels; }
public:
	GLTexture2D(const char* texturePath);
	GLTexture2D(glm::ivec2 size, int sizedInternalFormat, int mipMapLevels = 1);

	~GLTexture2D();

	void BindToUnit(GLuint unit);
	void Bind();

	void SetData(glm::vec3 pixels[], glm::ivec2 size);

	void DownloadTexture(std::vector<glm::vec3>& pixels);
};