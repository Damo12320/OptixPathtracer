#pragma once

#include "Microfacet.h"
#include "../Surface.h"
#include "../RayData.h"

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

        __device__ glm::vec3 f(Surface& surface, glm::vec3 wi) {
			if (wi.z == 0 || surface.outgoingRay.z == 0) return glm::vec3(0);
            if (!SpherGeom::SameHemisphere(wi, surface.outgoingRay)) return glm::vec3(0);
            if (surface.IsEffectifvelySmooth()) return glm::vec3(0);

            float cosTheta_o = SpherGeom::AbsCosTheta(surface.outgoingRay), cosTheta_i = SpherGeom::AbsCosTheta(wi);
            if (cosTheta_i == 0 || cosTheta_o == 0) return glm::vec3(0);
            glm::vec3 wm = surface.outgoingRay + wi;
            if (Sqr(glm::length(wm)) == 0) return glm::vec3(0);
            wm = glm::normalize(wm);

            //Fresnel
            //glm::vec3 F = FrComplex(AbsDot(wo, wm), eta, k);
			const glm::vec3 specularColor = glm::vec3(1);//white -> physically correct/ no influence
			glm::vec3 F = Fresnel82(specularColor, surface.albedo, AbsDot(surface.outgoingRay, wm));

            float alpha = Sqr(surface.roughness);

            return Microfacet::D_Isotropic(wm, alpha) * F * Microfacet::G_Isotropic(surface.outgoingRay, wi, alpha) /
                (4 * cosTheta_i * cosTheta_o);
        }

		__device__ bool Sample_f(unsigned int& randomSeed, Surface& surface, glm::vec3& wi, float& pdf, glm::vec3& sample) {
            const glm::vec3 specularColor = glm::vec3(1);//white -> physically correct/ no influence
            
            //if (!(sampleFlags & BxDFReflTransFlags::Reflection)) return {};
            if (surface.IsEffectifvelySmooth()) {
                wi = glm::vec3(-surface.outgoingRay.x, -surface.outgoingRay.y, surface.outgoingRay.z);
                //glm::vec3 wi(surface.outgoingRay.x, surface.outgoingRay.y, -surface.outgoingRay.z);

                sample = Fresnel82(specularColor, surface.albedo, SpherGeom::AbsCosTheta(wi)) / SpherGeom::AbsCosTheta(wi);
                //SampledSpectrum f = FrComplex(AbsCosTheta(wi), eta, k) / AbsCosTheta(wi);
                pdf = 1.0f;
                return true;
                //return BSDFSample(f, wi, 1, BxDFFlags::SpecularReflection);
            }

            float alpha = Sqr(surface.roughness);
            //float alpha = surface.roughness;

            glm::vec3 wm = Microfacet::Sample_wm(randomSeed, surface.outgoingRay, alpha);
            /*if (glm::dot(wm, surface.outgoingRay) < 0.0f) {
                wm = -wm;
            }*/

            wi = glm::reflect(-surface.outgoingRay, wm);
            //wi = -surface.outgoingRay + 2 * glm::dot(surface.outgoingRay, wm) * wm;

            //wi = glm::normalize(wi);

            //if (IsDebugRay() && glm::dot(wm, surface.sNormal) < 0.0f) {
            //    printf("wm not in same hemisphere as sNormal \n");
            //}
            //
            //if (IsDebugRay() && glm::dot(wi, surface.sNormal) < 0.0f) {
            //    printf("wi not in same hemisphere as sNormal \n");
            //}
            //
            //if (IsDebugRay() && glm::dot(wi, wm) < 0.0f) {
            //    printf("wi not in same hemisphere as wm \n");
            //}
            //
            //if (IsDebugRay() && glm::dot(wi, surface.outgoingRay) < 0.0f) {
            //    printf("wi not in same hemisphere as outgoingRay \n");
            //}
            //
            //if (IsDebugRay()) {
            //    Print("sNormal", surface.sNormal);
            //    Print("outgoingRay", surface.outgoingRay);
            //    Print("wm", wm);
            //    Print("wi", wi);
            //}

            /*if (glm::dot(wm, wi) < 0.0f) {
                wi = -wi;
            }*/

            if (!SpherGeom::SameHemisphere(surface.outgoingRay, wi)) {
                return false;
            }

            //<< Compute PDF of wi for microfacet reflection >>
            pdf = Microfacet::PDF_Isotropic(surface.outgoingRay, wm, alpha) / (4 * AbsDot(surface.outgoingRay, wm));

            float cosTheta_o = SpherGeom::AbsCosTheta(surface.outgoingRay);
            float cosTheta_i = SpherGeom::AbsCosTheta(wi);

            //Fresnel
            //SampledSpectrum F = FrComplex(AbsDot(wo, wm), eta, k);
            glm::vec3 F = Fresnel82(specularColor, surface.albedo, AbsDot(surface.outgoingRay, wm));

            glm::vec3 f = Microfacet::D_Isotropic(wm, alpha) * F * Microfacet::G_Isotropic(surface.outgoingRay, wi, alpha) / (4 * cosTheta_i * cosTheta_o);
            //return BSDFSample(f, wi, pdf, BxDFFlags::GlossyReflection);

            sample = f;

            return true;
		}

    }
}