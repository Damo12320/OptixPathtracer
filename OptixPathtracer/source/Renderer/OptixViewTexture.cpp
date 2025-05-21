#include "OptixViewTexture.h"

OptixViewTexture::OptixViewTexture() {
	glGenTextures(1, &this->handle);
}

OptixViewTexture::OptixViewTexture(uint32_t pixels[], glm::ivec2 size) {
	glGenTextures(1, &this->handle);
	glBindTexture(GL_TEXTURE_2D, this->handle);

	GLenum texFormat = GL_RGBA;
	GLenum texelType = GL_UNSIGNED_BYTE;

	glTexImage2D(GL_TEXTURE_2D, 0, texFormat, size.x, size.y, 0, GL_RGBA, texelType, pixels);
}

void OptixViewTexture::SetData(uint32_t pixels[], glm::ivec2 size) {
	glBindTexture(GL_TEXTURE_2D, this->handle);

	GLenum texFormat = GL_RGBA;
	GLenum texelType = GL_UNSIGNED_BYTE;

	glTexImage2D(GL_TEXTURE_2D, 0, texFormat, size.x, size.y, 0, GL_RGBA, texelType, pixels);

	glEnable(GL_TEXTURE_2D);
	//glBindTexture(GL_TEXTURE_2D, this->handle);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void OptixViewTexture::BindTexture() {
	glBindTexture(GL_TEXTURE_2D, this->handle);
}