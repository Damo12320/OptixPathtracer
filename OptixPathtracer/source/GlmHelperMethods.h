#pragma once

#include "3rdParty/glm/glm.hpp"

namespace GlmHelper {
	glm::vec3 BlenderToEnginePosition(glm::vec3 blenderPosition);
	glm::vec3 BlenderToEngineRotation(glm::vec3 blenderRotation);

	glm::vec3 BlenderToEnginePosition(float x, float y, float z);
	glm::vec3 BlenderToEngineRotation(float x, float y, float z);
}