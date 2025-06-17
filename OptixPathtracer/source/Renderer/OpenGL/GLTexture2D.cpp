#include "GLTexture2D.h"

GLTexture2D::GLTexture2D(const char* texturePath) {
	glGenTextures(1, &this->ID);
	glBindTexture(GL_TEXTURE_2D, this->ID);

	// set the texture wrapping/filtering options (on the currently bound texture object)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// load and generate the texture
	int width, height, nrChannels;
	unsigned char* data = stbi_load(texturePath, &width, &height, &nrChannels, 0);
	if (data)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else
	{
		std::cout << "ERROR::Failed to load texture" << std::endl;
	}

	this->width = width;
	this->height = height;
	stbi_image_free(data);
}

GLTexture2D::GLTexture2D(glm::ivec2 size, int sizedInternalFormat, int mipMapLevels) {
	glCreateTextures(GL_TEXTURE_2D, 1, &this->ID);

	if (mipMapLevels == -1)
	{
		mipMapLevels = std::log2f(std::floor(std::max(1, std::max(size.x, size.y)))) + 1;
	}

	glTextureStorage2D(this->ID, mipMapLevels, sizedInternalFormat, size.x, size.y);

	glTextureParameteri(this->ID, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTextureParameteri(this->ID, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	this->width = size.x;
	this->height = size.y;
}





GLTexture2D::~GLTexture2D() {
	std::cout << "TEXTURE2D::Deleting" << std::endl;
	glDeleteTextures(1, &this->ID);
}

void GLTexture2D::BindToUnit(GLuint unit) {
	glBindTextureUnit(unit, this->ID);
}

void GLTexture2D::Bind() {
	glBindTexture(GL_TEXTURE_2D, this->ID);
}

void GLTexture2D::SetData(glm::vec3 pixels[], glm::ivec2 size) {
	this->width = size.x;
	this->height = size.y;

	glTextureSubImage2D(this->ID, 0, 0, 0, size.x, size.y, GL_RGB, GL_FLOAT, pixels);
}