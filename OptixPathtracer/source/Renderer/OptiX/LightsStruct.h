#pragma once

#include <cuda_runtime.h>
#include "../../3rdParty/glm/glm.hpp"

struct PointLight
{
	glm::vec3 position;
	glm::vec3 color;
};

//__device__ float DistanceSquared(glm::vec3 vector) {
//	return vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;;
//}
//
//__device__ float DistanceSquared(glm::vec3 vector, glm::vec3 lightPosition) {
//	return DistanceSquared(lightPosition - vector);
//}
//
//
//struct PointLight
//{
//	glm::vec3 position;
//	glm::vec3 color;
//
//	__device__ float GetPDF() {
//		return 1.0f;
//	}
//
//	__device__ glm::vec3 GetSample(glm::vec3 surfacePosition) {
//		return color / DistanceSquared(this->position, surfacePosition);
//	}
//};