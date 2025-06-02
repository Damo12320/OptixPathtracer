#pragma once

#include "GLTexture2D.h"
#include <map>

class Framebuffer {
private:
	unsigned int ID;

	glm::ivec2 size;

	std::map<GLenum, std::unique_ptr<GLTexture2D>> attachedTextures;
public:
	Framebuffer();

	~Framebuffer();

	void AttachNewTexture2D(GLenum attachment, glm::ivec2 size, int sizedInternalFormat = GL_RGBA8, int filter = GL_NEAREST, int mipMapLevels = -1);

	GLTexture2D* GetAttachedTexture(GLenum attachment);

	void Bind();

	unsigned int GetID();
	glm::ivec2 GetSize();

	bool IsComplete();
};