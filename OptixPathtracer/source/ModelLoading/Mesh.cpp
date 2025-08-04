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

void Mesh::CalculateTangentBasis() {
	this->tangents = std::vector<glm::vec3>();
	this->bitangents = std::vector<glm::vec3>();

	this->tangents.resize(this->vertecies.size());
	this->bitangents.resize(this->vertecies.size());

	if (this->texCoord.size() != this->vertecies.size()) {
		this->texCoord.resize(this->vertecies.size());
		return;
	}

	for (int i = 0; i < this->index.size(); i++)
	{
		glm::vec3 v0 = this->vertecies[this->index[i].x];
		glm::vec3 v1 = this->vertecies[this->index[i].y];
		glm::vec3 v2 = this->vertecies[this->index[i].z];

		glm::vec2 uv0 = this->texCoord[this->index[i].x];
		glm::vec2 uv1 = this->texCoord[this->index[i].y];
		glm::vec2 uv2 = this->texCoord[this->index[i].z];

		glm::vec3 deltaPos1 = v1 - v0;
		glm::vec3 deltaPos2 = v2 - v0;

		glm::vec2 deltaUV1 = uv1 - uv0;
		glm::vec2 deltaUV2 = uv2 - uv0;

		float r = 1.0 / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
		glm::vec3 tangent = (deltaPos1 * deltaUV2.y - deltaPos2 * deltaUV1.y) * r;
		glm::vec3 bitangent = (deltaPos2 * deltaUV1.x - deltaPos1 * deltaUV2.x) * r;

		tangent = glm::normalize(tangent);
		bitangent = glm::normalize(bitangent);


		this->tangents[this->index[i].x] = tangent;
		this->tangents[this->index[i].y] = tangent;
		this->tangents[this->index[i].z] = tangent;

		this->bitangents[this->index[i].x] = bitangent;
		this->bitangents[this->index[i].y] = bitangent;
		this->bitangents[this->index[i].z] = bitangent;
	}
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