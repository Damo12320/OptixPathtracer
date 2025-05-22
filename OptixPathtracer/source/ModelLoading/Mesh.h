#pragma once
#include "../3rdParty/glm/glm.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include "../3rdParty/glm/gtc/quaternion.hpp"
#include "../3rdParty/glm/gtx/quaternion.hpp"
#include <vector>
#include <string>

class Mesh {
public:
	std::string meshName;

	std::vector<glm::vec3> vertecies;
	std::vector<glm::vec3> normal;
	std::vector<glm::vec2> texCoord;

	std::vector<glm::ivec3> index;

	//Transformation
	glm::vec3 translation;
	glm::vec3 scale;
	glm::quat rotation;

	std::vector<float> ModelMatrix;

	//Material
	glm::vec3 albedo;
	float metallic;
	float roughness;

	//Textures
	int albedoTex;
	int normalTex;
	int metalRoughTex;

public:
	glm::mat4x4 GetModelMatrix();
	void CalculateModelMatrix();

	bool HasAlbedoTex();
	bool HasNormalTex();
	bool HasMetalRoughTex();
};