#include "Framebuffer.h"

Framebuffer::Framebuffer() {
	glGenFramebuffers(1, &this->ID);

	glBindFramebuffer(GL_FRAMEBUFFER, this->ID);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

Framebuffer::~Framebuffer() {
	std::cout << "FRAMEBUFFER::Deleting" << std::endl;
	glDeleteFramebuffers(1, &this->ID);
}




void Framebuffer::AttachNewTexture2D(GLenum attachment, glm::ivec2 size, int sizedInternalFormat, int filter, int mipMapLevels) {
	if (this->attachedTextures.size() > 0) {
		GLTexture2D* firstTexture = this->attachedTextures.begin()->second.get();
		if (firstTexture->GetWidth() != size.x || firstTexture->GetHeight() != size.y) {
			std::cout << "ERROR::FRAMEBUFFER::ATTACHMENT:: The Texturedimensions " << size.x << "|" << size.y <<
				" dosen't match the first inserted Dimensions: " << firstTexture->GetWidth() << "|" << firstTexture->GetHeight() << std::endl;
			return;
		}

		if (this->attachedTextures.find(attachment) != this->attachedTextures.end()) {
			std::cout << "ERROR::FRAMEBUFFER::ATTACHMENT:: Attachment " << attachment << " already exists!" << std::endl;
			return;
		}
	}

	this->size = size;

	std::unique_ptr<GLTexture2D> texture = std::make_unique<GLTexture2D>(size, sizedInternalFormat, mipMapLevels);

	glBindTexture(GL_TEXTURE_2D, texture->GetID());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);

	glBindFramebuffer(GL_FRAMEBUFFER, this->ID);
	//glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture->GetID(), 0);
	glNamedFramebufferTexture(this->ID, attachment, texture->GetID(), 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	this->attachedTextures[attachment] = std::move(texture);
}

GLTexture2D* Framebuffer::GetAttachedTexture(GLenum attachment) {
	if (this->attachedTextures.find(attachment) == this->attachedTextures.end()) {
		std::cout << "ERROR::FRAMEBUFFER::ATTACHMENT:: Attachment " << attachment << " does not exist!" << std::endl;
		return nullptr;
	}

	return this->attachedTextures[attachment].get();
}

void Framebuffer::Bind() {
	glBindFramebuffer(GL_FRAMEBUFFER, this->ID);
}

unsigned int Framebuffer::GetID() {
	return this->ID;
}

glm::ivec2 Framebuffer::GetSize() {
	return this->size;
}

bool Framebuffer::IsComplete() {
	glBindFramebuffer(GL_FRAMEBUFFER, this->ID);
	bool complete = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	return complete;
}