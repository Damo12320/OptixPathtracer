#pragma once

#include "Microfacet.h"
#include "../Surface.h"

namespace ConductorBRDF {
	__device__ glm::vec3 FresnelSchlick(glm::vec3 F0, float mu) {
		return F0 + (glm::vec3(1.0) - F0) * powf(1 - mu, 5);
	}


	/**
	* The Fresnel 82 after the OpenPBR (for Conductors)
	*
	* @param glm::vec3 specularColor = The tint the fresnel has
	* @param glm::vec3 F0 = The BaseColor
	* @param float mu = the cosine of the incident angle
	*
	*/
	__device__ glm::vec3 Fresnel82(glm::vec3 specularColor, glm::vec3 F0, glm::vec3 microNormal, glm::vec3 direction) {
		const float mu = Microfacet::AbsCosTheta(microNormal, direction);

		const float mu_ = 1.0f / 7.0f;

		glm::vec3 Fmu = FresnelSchlick(F0, mu);
		glm::vec3 Fmu_ = FresnelSchlick(F0, mu_);

		float middle = mu * powf(1 - mu, 6);
		middle /= mu_ * pow(1.0f - mu_, 6.0f);

		glm::vec3 color = specularColor * Fmu_;

		return Fmu - middle * (Fmu_ - color);
	}

	__device__ glm::vec3 ConductorBRDF(Surface& surface, glm::vec3 outgoing) {
		const glm::vec3 specularTint = glm::vec3(1.0);


		if (surface.IsEffectifvelySmooth()) {
			glm::vec3 fresnel = Fresnel82(specularTint, surface.albedo, surface.sNormal, outgoing);

			return fresnel / Microfacet::AbsCosTheta(surface.sNormal, surface.incommingRay);
		}


		glm::vec3 microfacetNormal;
		if (!Microfacet::MicrofacetNormal(surface.incommingRay, outgoing, microfacetNormal)) {
			return glm::vec3(0);
		}

		const float cosTheta_i = Microfacet::CosTheta(surface.sNormal, surface.incommingRay);//AbsCosTheta?
		const float cosTheta_o = Microfacet::CosTheta(surface.sNormal, outgoing);

		if (cosTheta_i == 0 || cosTheta_o == 0) return glm::vec3(0);

		glm::vec3 fresnel = Fresnel82(specularTint, surface.albedo, surface.sNormal, outgoing);
		//printf("FRESNEL %f, %f, %f \n", fresnel.x, fresnel.y, fresnel.z);


		//printf("cosTheta %f, %f \n", cosTheta_i, cosTheta_o);

		float alpha = surface.roughness * surface.roughness;
		float alpha2 = alpha * alpha;

		float NDF = Microfacet::D_IsotropicMicrofacetDistribution(alpha2, surface.sNormal, microfacetNormal);

		float MSF = Microfacet::G_MSF(surface.sNormal, surface.incommingRay, outgoing, alpha2);

		return ( NDF * fresnel * MSF ) / ( 4.0f * cosTheta_i * cosTheta_o);
	}

	__device__ float GetPDF(Surface& surface, glm::vec3 outgoing) {
		if (surface.IsEffectifvelySmooth()) return 0;

		glm::vec3 microfacetNormal;
		if (Microfacet::MicrofacetNormal(surface.incommingRay, outgoing, microfacetNormal)) {
			return 0;
		}

		glm::faceforward(microfacetNormal, surface.sNormal, microfacetNormal);

		float alpha = surface.roughness * surface.roughness;
		float alpha2 = alpha * alpha;

		return Microfacet::PDF(surface.sNormal, alpha2, outgoing, microfacetNormal) / (4 * fabsf(glm::dot(outgoing, microfacetNormal)));
	}
}