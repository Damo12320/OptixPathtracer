#include "GlmHelperMethods.h"

namespace GlmHelper {
	glm::vec3 BlenderToEnginePosition(glm::vec3 blenderPosition) {
		return glm::vec3(blenderPosition.x, blenderPosition.z, -blenderPosition.y);
	}

	glm::vec3 BlenderToEngineRotation(glm::vec3 blenderRotation) {
		return glm::vec3(90 - blenderRotation.x, 180 + blenderRotation.z, blenderRotation.y);
	}

	glm::vec3 BlenderToEnginePosition(float x, float y, float z) {
		return glm::vec3(x, z, -y);
	}

	glm::vec3 BlenderToEngineRotation(float x, float y, float z) {
		return glm::vec3(90 - x, 180 + z, y);
	}
}