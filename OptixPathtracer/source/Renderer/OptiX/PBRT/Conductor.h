#pragma once

#include"Microfacet.h"
#include"../Surface.h"

namespace PBRT {
    namespace Conductor {
#pragma region OpenPBR Fresnel

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
		__device__ glm::vec3 Fresnel82(glm::vec3 specularColor, glm::vec3 F0, float absCosTheta) {
			const float mu = absCosTheta;

			const float mu_ = 1.0f / 7.0f;

			glm::vec3 Fmu = FresnelSchlick(F0, mu);
			glm::vec3 Fmu_ = FresnelSchlick(F0, mu_);

			float middle = mu * powf(1 - mu, 6);
			middle /= mu_ * pow(1.0f - mu_, 6.0f);

			glm::vec3 color = specularColor * Fmu_;

			return Fmu - middle * (Fmu_ - color);
		}


#pragma endregion

        __device__ glm::vec3 f(Surface& surface, glm::vec3 wo) {
			if (wo.z == 0 || surface.incommingRay.z == 0) return glm::vec3(0);
            if (!SpherGeom::SameHemisphere(wo, surface.incommingRay)) return glm::vec3(0);
            if (surface.IsEffectifvelySmooth()) return glm::vec3(0);

            float cosTheta_o = SpherGeom::AbsCosTheta(surface.incommingRay), cosTheta_i = SpherGeom::AbsCosTheta(wo);
            if (cosTheta_i == 0 || cosTheta_o == 0) return {};
            glm::vec3 wm = surface.incommingRay + wo;
            if (Sqr(glm::length(wm)) == 0) return {};
            wm = glm::normalize(wm);

            //Fresnel
            //glm::vec3 F = FrComplex(AbsDot(wo, wm), eta, k);
			const glm::vec3 specularColor = glm::vec3(1);//white -> physically correct/ no influence
			glm::vec3 F = Fresnel82(specularColor, surface.albedo, AbsDot(wo, wm));

            float alpha = Sqr(surface.roughness);

            return Microfacet::D_Isotropic(wm, alpha) * F * Microfacet::G_Isotropic(surface.incommingRay, wo, alpha) /
                (4 * cosTheta_i * cosTheta_o);
        }

		__device__ bool Sample_f(unsigned int& randomSeed, Surface& surface, glm::vec3& wo, float& pdf, glm::vec3& sample) {
            const glm::vec3 specularColor = glm::vec3(1);//white -> physically correct/ no influence
            
            //if (!(sampleFlags & BxDFReflTransFlags::Reflection)) return {};
            if (surface.IsEffectifvelySmooth()) {
                glm::vec3 wi(-surface.incommingRay.x, -surface.incommingRay.y, surface.incommingRay.z);

                sample = Fresnel82(specularColor, surface.albedo, SpherGeom::AbsCosTheta(wi)) / SpherGeom::AbsCosTheta(wi);
                //SampledSpectrum f = FrComplex(AbsCosTheta(wi), eta, k) / AbsCosTheta(wi);
                wo = wi;
                pdf = 1;
                return true;
                //return BSDFSample(f, wi, 1, BxDFFlags::SpecularReflection);
            }

            float alpha = Sqr(surface.roughness);

            glm::vec3 wm = Microfacet::Sample_wm(randomSeed, surface.incommingRay, alpha);
            glm::vec3 wi = glm::reflect(surface.incommingRay, wm);
            if (!SpherGeom::SameHemisphere(surface.incommingRay, wi)) return false;

            //<< Compute PDF of wi for microfacet reflection >>
            pdf = Microfacet::PDF_Isotropic(surface.incommingRay, wm, alpha) / (4 * AbsDot(surface.incommingRay, wm));

            float cosTheta_o = SpherGeom::AbsCosTheta(surface.incommingRay), cosTheta_i = SpherGeom::AbsCosTheta(wi);

            //Fresnel
            //SampledSpectrum F = FrComplex(AbsDot(wo, wm), eta, k);
            glm::vec3 F = Fresnel82(specularColor, surface.albedo, AbsDot(surface.incommingRay, wm));

            glm::vec3 f = Microfacet::D_Isotropic(wm, alpha) * F * Microfacet::G_Isotropic(surface.incommingRay, wi, alpha) / (4 * cosTheta_i * cosTheta_o);
            //return BSDFSample(f, wi, pdf, BxDFFlags::GlossyReflection);

            sample = f;

            return true;
		}

    }
}