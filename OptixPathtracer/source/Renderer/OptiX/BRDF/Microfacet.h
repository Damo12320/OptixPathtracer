#pragma once

#include "../glmCUDA.h"

namespace Microfacet {
#pragma region OpenPBR

	//OpenPBR https://academysoftwarefoundation.github.io/OpenPBR/#mjx-eqn-GGX
	__device__ float NDF_Dggx(float roughness, glm::vec3 surfaceNormal, glm::vec3 microNormal) {
		float cosTheta = glm::dot(surfaceNormal, microNormal);

		float alpha = roughness * roughness;
		float alpha2 = alpha * alpha;

		float tan2 = tanf(cosTheta) * tanf(cosTheta);

		float result = 1 + (tan2 / alpha2);
		return powf(result, -2);
	}


#pragma endregion

#pragma region RaytracingBook
	// Phi: Rotation of vector around the normal (only relevant for anisotropic distribution)
	__device__ float CosPhi(glm::vec3 surfacetangent, glm::vec3 surfaceNormal, glm::vec3 microNormal) {
		glm::vec3 p = microNormal - glm::dot(surfaceNormal, microNormal) * surfaceNormal;
		p = glm::normalize(p);

		return glm::dot(surfacetangent, p);
	}
	__device__ float Cos2Phi(glm::vec3 surfacetangent, glm::vec3 surfaceNormal, glm::vec3 microNormal) {
		float cosPhi = CosPhi(surfacetangent, surfaceNormal, microNormal);
		return cosPhi * cosPhi;
	}

	__device__ float SinPhi(glm::vec3 surfacebitangent, glm::vec3 surfaceNormal, glm::vec3 microNormal) {
		glm::vec3 p = microNormal - glm::dot(surfaceNormal, microNormal) * surfaceNormal;
		p = glm::normalize(p);

		return glm::dot(surfacebitangent, p);
	}
	__device__ float Sin2Phi(glm::vec3 surfacebitangent, glm::vec3 surfaceNormal, glm::vec3 microNormal) {
		float sinPhi = SinPhi(surfacebitangent, surfaceNormal, microNormal);
		return sinPhi * sinPhi;
	}

	// Theta: Angle between the vector and normal
	__device__ float CosTheta(glm::vec3 surfaceNormal, glm::vec3 microNormal) {
		return glm::max(glm::dot(surfaceNormal, microNormal), 0.0f);
	}
	__device__ float Cos2Theta(glm::vec3 surfaceNormal, glm::vec3 microNormal) {
		float cosTheta = CosTheta(surfaceNormal, microNormal);
		return glm::dot(surfaceNormal, microNormal);
	}
	__device__ float AbsCosTheta(glm::vec3 surfaceNormal, glm::vec3 microNormal) {
		return fabsf(glm::dot(surfaceNormal, microNormal));
	}

	__device__ float Tan2Theta(float cos2Theta) {
		return (1 - cos2Theta) / cos2Theta;
	}

	__device__ float D_AnisotropicMicrofacetDistribution(glm::vec2 alpha, glm::vec3 sNormal, glm::vec3 sTangent, glm::vec3 sBitangent, glm::vec3 microNormal) {
		// Gather main ingredients
		const float cos2Theta = Cos2Theta(sNormal, microNormal);
		const float tan2Theta = Tan2Theta(cos2Theta);

		const float cos2Phi = Cos2Phi(sTangent, sNormal, microNormal);
		const float sin2Phi = Sin2Phi(sBitangent, sNormal, microNormal);

		glm::vec2 alpha2 = alpha * alpha;

		// Get main fractions
		float fraction1 = cos2Phi / alpha2.x;
		float fraction2 = sin2Phi / alpha2.y;

		// The actual Distribution Term
		float mainTerm = 1 + tan2Theta * (fraction1 + fraction2);
		mainTerm = mainTerm * mainTerm;

		// The normaliing Term
		const float pi = 3.14159265359;
		float normalizing = pi * alpha.x * alpha.y * cos2Theta * cos2Theta;

		return 1 / (normalizing * mainTerm);
	}

	__device__ float D_IsotropicMicrofacetDistribution(float alpha2, glm::vec3 sNormal, glm::vec3 microNormal) {
		// Gather main ingredients
		const float cos2Theta = Cos2Theta(sNormal, microNormal);
		const float tan2Theta = Tan2Theta(cos2Theta);

		// The actual Distribution Term
		float mainTerm = 1 + (tan2Theta / alpha2);
		mainTerm = mainTerm * mainTerm;

		// The normaliing Term
		const float pi = 3.14159265359;
		float normalizing = pi * alpha2 * cos2Theta * cos2Theta;

		return 1 / (normalizing * mainTerm);
	}


	__device__ float Lambda(glm::vec3 sNormal, glm::vec3 direction, float alpha2) {
		const float cos2Theta = Cos2Theta(sNormal, direction);
		const float tan2Theta = Tan2Theta(cos2Theta);

		float numerator = sqrtf(1 + alpha2 * tan2Theta) - 1;
		 
		return numerator / 2;
	}
	__device__ float G_MSF(glm::vec3 sNormal, glm::vec3 wi, glm::vec3 wo, float alpha2) {
		return 1 / (1 + Lambda(sNormal, wo, alpha2) + Lambda(sNormal, wi, alpha2));
	}
	__device__ float G1(glm::vec3 sNormal, glm::vec3 w, float alpha2){
		return 1 / (1 + Lambda(sNormal, w, alpha2));
	}

	__device__ bool MicrofacetNormal(glm::vec3 wi, glm::vec3 wo, glm::vec3& microfacetNormal) {
		glm::vec3 wm = wi + wo;
		if (glm::length(wm) == 0) {
			return false;
		}

		microfacetNormal = glm::normalize(wm);

		return true;
	}

	__device__ float D(glm::vec3 sNormal, float alpha2, glm::vec3 w, glm::vec3 wm) {
		return G1(sNormal, w, alpha2) / ( AbsCosTheta(sNormal, w) * D_IsotropicMicrofacetDistribution(alpha2, sNormal, wm) * fabsf(glm::dot(w, wm)) );
	}

	__device__ float PDF(glm::vec3 sNormal, float alpha2, glm::vec3 w, glm::vec3 wm) {
		D(sNormal, alpha2, w, wm);
	}
#pragma endregion

}