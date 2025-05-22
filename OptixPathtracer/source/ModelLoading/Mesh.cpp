#include "Mesh.h"

#include "../3rdParty/glm/gtc/type_ptr.hpp"
#include <iostream>

glm::mat4x4 Mesh::GetModelMatrix() {
	//glm::mat4x4 matrix = glm::identity<glm::mat4x4>();

	glm::mat4x4 translation = glm::identity<glm::mat4x4>();
	translation = glm::translate(translation, this->translation);

	glm::mat4x4 rotation = glm::toMat4(this->rotation);

	glm::mat4x4 scale = glm::identity<glm::mat4x4>();
	scale = glm::scale(scale, this->scale);


	//matrix = glm::translate(matrix, this->translation);
	//matrix *= glm::toMat4(this->rotation);
	//matrix = glm::scale(matrix, this->scale);

	return translation * rotation * scale;
}

void Mesh::CalculateModelMatrix() {
	glm::mat4x4 matrix = this->GetModelMatrix();

	matrix = glm::transpose(matrix);

	const float* matrixArray = glm::value_ptr(matrix);

	for (int i = 0; i < 4*4; i++) {
		this->ModelMatrix.push_back(matrixArray[i]);
	}

	/*this->ModelMatrix.erase(this->ModelMatrix.end() - 1);
	this->ModelMatrix.erase(this->ModelMatrix.end() - 1);
	this->ModelMatrix.erase(this->ModelMatrix.end() - 1);
	this->ModelMatrix.erase(this->ModelMatrix.end() - 1);*/
}

bool Mesh::HasAlbedoTex() {
	return this->albedoTex >= 0;
}

bool Mesh::HasNormalTex() {
	return this->normalTex >= 0;
}

bool Mesh::HasMetalRoughTex() {
	return this->metalRoughTex >= 0;
}