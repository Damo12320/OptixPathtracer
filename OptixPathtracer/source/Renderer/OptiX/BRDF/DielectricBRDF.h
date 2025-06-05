#pragma once

#include "../glmCUDA.h"
#include "../Surface.h"

namespace DielectricBRDF {
	__device__ glm::vec3 DielectricBRDF(Surface& surface, glm::vec3 outgoing) {
		return surface.albedo / 3.14159265359f;
	}
}