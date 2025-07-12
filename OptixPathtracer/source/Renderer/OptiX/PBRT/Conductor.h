#pragma once

#include "Microfacet.h"
#include "../Surface.h"
#include "BSDFSample.h"
#include "Complex.h"
#include "../RayData.h"

namespace PBRT {
    namespace Conductor {
#pragma region OpenPBR Fresnel

		__device__ __host__ glm::vec3 FresnelSchlick(glm::vec3 F0, float mu) {
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
        //https://academysoftwarefoundation.github.io/OpenPBR/#model/basesubstrate/glossy-diffuse
		__device__ __host__ glm::vec3 Fresnel82(glm::vec3 specularColor, glm::vec3 F0, float absCosTheta) {
			const float mu = absCosTheta;

			const float mu_ = 1.0f / 7.0f;

			glm::vec3 Fmu = FresnelSchlick(F0, mu);
			glm::vec3 Fmu_ = FresnelSchlick(F0, mu_);

			float middle = mu * powf(1 - mu, 6);
			middle /= mu_ * pow(1.0f - mu_, 6.0f);

			glm::vec3 color = specularColor * Fmu_;

			return Fmu - middle * (Fmu_ - color);
		}

        __device__ __host__ float FrComplex(float cosTheta_i, Complex::complex eta) {
            cosTheta_i = glm::clamp(cosTheta_i, 0.0f, 1.0f);
            // Compute complex $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
            float sin2Theta_i = 1.0f - Sqr(cosTheta_i);
            Complex::complex sin2Theta_t = sin2Theta_i / Sqr(eta);
            Complex::complex cosTheta_t = Complex::sqrt(1.0f - sin2Theta_t);

            Complex::complex r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
            Complex::complex r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
            return (Complex::norm(r_parl) + Complex::norm(r_perp)) / 2.0f;
        }

        __device__ __host__ glm::vec3 FresnelComplex(float cosTheta_i, glm::vec3 reflectance) {
            //if (eta) {
            //    etas = texEval(eta, ctx, lambda);
            //    ks = texEval(k, ctx, lambda);
            //}
            //else {
            //    // Avoid r==0 NaN case...
            //    SampledSpectrum r = Clamp(texEval(reflectance, ctx, lambda), 0, .9999);
            //    etas = SampledSpectrum(1.f);
            //    ks = 2 * Sqrt(r) / Sqrt(ClampZero(SampledSpectrum(1) - r));
            //}

            glm::vec3 eta, k;

            glm::vec3 r = SaveClamp(reflectance, 0.0f, 0.9999f);
            eta = glm::vec3(1.0f);
            k = 2.0f * SaveSqrt(r) / SaveSqrt(SaveMax(glm::vec3(1) - r, 0.0f));

            //Copper (https://chris.hindefjord.se/resources/rgb-ior-metals/)
            //eta = glm::vec3(0.27105f, 0.67693f, 1.31640f);
            //k = glm::vec3(3.60920f, 2.62480f, 2.29210f);

            //Blender Default
            //eta = glm::vec3(2.757f, 2.513f, 2.231f);
            //k = glm::vec3(3.867f, 3.404f, 3.009f);


            glm::vec3 result;

            result.r = FrComplex(cosTheta_i, Complex::complex(eta.r, k.r));
            result.g = FrComplex(cosTheta_i, Complex::complex(eta.g, k.g));
            result.b = FrComplex(cosTheta_i, Complex::complex(eta.b, k.b));

            /*for (int i = 0; i < 3; i++) {
                result[i] = FrComplex(cosTheta_i, Complex::complex(eta[i], k[i]));
            }*/

            return result;
        }


#pragma endregion

        __device__ __host__ glm::vec3 f(glm::vec3 surfaceColor, float roughness, glm::vec3 wo, glm::vec3 wi) {
            float alpha = Surface::GetAlpha(roughness);

            if (!SpherGeom::SameHemisphere(wo, wi))
                return {};
            if (Surface::IsEffectifvelySmooth(alpha))
                return {};
            // Evaluate rough conductor BRDF
            // Compute cosines and $\wm$ for conductor BRDF
            float cosTheta_o = SpherGeom::AbsCosTheta(wo);
            float cosTheta_i = SpherGeom::AbsCosTheta(wi);
            if (cosTheta_i == 0 || cosTheta_o == 0)
                return {};

            glm::vec3 wm = wi + wo;
            if (LengthSqr(wm) == 0)
                return {};
            wm = glm::normalize(wm);

            // Evaluate Fresnel factor _F_ for conductor BRDF
            glm::vec3 F = FresnelComplex(AbsDot(wo, wm), surfaceColor);

            return Microfacet::D_Isotropic(wm, alpha) * F * Microfacet::G_Isotropic(wo, wi, alpha) / (4.0f * cosTheta_i * cosTheta_o);
        }

        __device__ __host__ bool Sample_f(unsigned int& randomSeed, glm::vec3 surfaceColor, float roughness, glm::vec3 wo, BSDFSample& sample) {
            //if (!(sampleFlags & BxDFReflTransFlags::Reflection))
                //return {};
            float alpha = Surface::GetAlpha(roughness);
            if (Surface::IsEffectifvelySmooth(alpha)) {
                // Sample perfect specular conductor BRDF
                glm::vec3 wi(-wo.x, -wo.y, wo.z);
                glm::vec3 f = FresnelComplex(SpherGeom::AbsCosTheta(wi), surfaceColor) / SpherGeom::AbsCosTheta(wi);
                //return BSDFSample(f, wi, 1, BxDFFlags::SpecularReflection);

                sample.color = f;
                sample.direction = wi;
                sample.pdf = 1.0f;

                sample.reflection = true;
                sample.transmission = false;
                sample.specular = true;
                sample.glossy = false;

                return true;
            }
            if (IsDebugRay()) printf("1 \n");

            // Sample rough conductor BRDF
            // Sample microfacet normal $\wm$ and reflected direction $\wi$
            if (wo.z == 0.0f)
                return false;
            glm::vec3 wm = Microfacet::Sample_wm(randomSeed, wo, alpha);
            //if (wm.z < 0) wm.z = -wm.z;

            if (IsDebugRay()) printf("2 \n");

            auto Reflect = [](glm::vec3 wo, glm::vec3 n) {
                return -wo + 2 * glm::dot(wo, n) * n;
                };

            glm::vec3 wi = Reflect(wo, wm);
            //glm::vec3 wi = glm::reflect(-wo, wm);
            if (!SpherGeom::SameHemisphere(wo, wi))
                return false;

            if (IsDebugRay()) printf("3 \n");

            // Compute PDF of _wi_ for microfacet reflection
            float pdf = Microfacet::PDF_Isotropic(wo, wm, alpha) / (4 * AbsDot(wo, wm));

            float cosTheta_o = SpherGeom::AbsCosTheta(wo);
            float cosTheta_i = SpherGeom::AbsCosTheta(wi);
            if (cosTheta_i == 0 || cosTheta_o == 0)
                return false;
            if (IsDebugRay()) printf("4 \n");

            // Evaluate Fresnel factor _F_ for conductor BRDF
            glm::vec3 F = FresnelComplex(AbsDot(wo, wm), surfaceColor);

            glm::vec3 f = Microfacet::D_Isotropic(wm, alpha) * F * Microfacet::G_Isotropic(wo, wi, alpha) / (4.0f * cosTheta_i * cosTheta_o);
            //return BSDFSample(f, wi, pdf, BxDFFlags::GlossyReflection);

            sample.color = f;
            sample.direction = wi;
            sample.pdf = pdf;

            sample.reflection = true;
            sample.transmission = false;
            sample.specular = false;
            sample.glossy = true;

            return true;
        }
    }
}