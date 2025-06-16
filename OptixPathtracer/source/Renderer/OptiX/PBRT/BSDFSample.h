#pragma once

#include "../glmCUDA.h"

struct BSDFSample {
	glm::vec3 color;
	float pdf;

	glm::vec3 direction;

	bool transmission;
	bool reflection;
	bool specular;
	bool glossy;
};