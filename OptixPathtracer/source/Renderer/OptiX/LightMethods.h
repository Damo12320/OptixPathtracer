#pragma once

#include <cuda_runtime.h>
#include "../../3rdParty/glm/glm.hpp"
#include "LightsStruct.h"
#include "random.h"

namespace Lighting {
	__device__ float DistanceSquared(glm::vec3 vector) {
		return vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;;
	}

	__device__ float DistanceSquared(glm::vec3 vector, glm::vec3 lightPosition) {
		return DistanceSquared(lightPosition - vector);
	}

	__device__ float GetPDF(PointLight* light) {
		return 1.0f;
	}

	__device__ glm::vec3 GetSample(glm::vec3 surfacePosition, PointLight* light) {
		return light->color / DistanceSquared(light->position, surfacePosition);
	}

	__device__ PointLight* GetRandomPointLight(LaunchParams* launchParams, unsigned int& randomSeed, float& propability) {
		if (launchParams->pointLightCount == 1) {
			propability = 1.0f;
			return &launchParams->pointlights[0];
		}

		if (launchParams->pointLightCount == 0) {
			propability = 0.0f;
			return {};
		}

		float rnd = RandomOptix::rnd(randomSeed);
		int index = rnd * launchParams->pointLightCount;

		propability = 1.0f / launchParams->pointLightCount;
		return &launchParams->pointlights[index];
	}
}