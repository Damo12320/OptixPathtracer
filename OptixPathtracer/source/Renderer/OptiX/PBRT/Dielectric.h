#pragma once

#include "Microfacet.h"
#include "../Surface.h"
#include "BSDFSample.h"

namespace PBRT {
    namespace Dielectric {

        __device__ __host__ float FresnelDielectric(float cosTheta_i, float ior = 1.5f) {
            //ior = 1.5f;

            cosTheta_i = glm::clamp(cosTheta_i, -1.0f, 1.0f);
            //<< Potentially flip interface orientation for Fresnel equations >>
            if (cosTheta_i < 0.0f) {
                ior = 1.0f / ior;
                cosTheta_i = -cosTheta_i;
            }

            //<< Compute for Fresnel equations using Snell’s law >>
            float sin2Theta_i = 1.0f - Sqr(cosTheta_i);
            float sin2Theta_t = sin2Theta_i / Sqr(ior);
            if (sin2Theta_t >= 1.0f)
                return 1.0f;
            float cosTheta_t = sqrtf(1.0f - sin2Theta_t);

            float r_parl = (ior * cosTheta_i - cosTheta_t) /
                (ior * cosTheta_i + cosTheta_t);
            float r_perp = (cosTheta_i - ior * cosTheta_t) /
                (cosTheta_i + ior * cosTheta_t);
            return (Sqr(r_parl) + Sqr(r_perp)) / 2.0f;
        }

        //__device__ __host__ bool Refract(glm::vec3 wi, glm::vec3 n, float eta, float* etap, glm::vec3* wt) {
        //    float cosTheta_i = glm::dot(n, wi);
        //    //<< Potentially flip interface orientation for Snell’s law >>
        //    if (cosTheta_i < 0.0f) {
        //        eta = 1 / eta;
        //        cosTheta_i = -cosTheta_i;
        //        n = -n;
        //    }

        //    //<< Compute using Snell’s law >>
        //    float sin2Theta_i = glm::max(0.0f, 1.0f - Sqr(cosTheta_i));
        //    float sin2Theta_t = sin2Theta_i / Sqr(eta);
        //    //<< Handle total internal reflection case >>
        //    if (sin2Theta_t >= 1.0f) return false;

        //    float cosTheta_t = sqrt(1 - sin2Theta_t);

        //    *wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * n;
        //    //<< Provide relative IOR along ray to caller >>
        //    if (etap) *etap = eta;

        //    return true;
        //}

        __device__ __host__ bool Refract(glm::vec3 wi, glm::vec3 n, float eta, float* etap, glm::vec3* wt) {
            float cosTheta_i = glm::dot(n, wi);
            // Potentially flip interface orientation for Snell's law
            if (cosTheta_i < 0.0f) {
                eta = 1 / eta;
                cosTheta_i = -cosTheta_i;
                n = -n;
            }

            // Compute $\cos\,\theta_\roman{t}$ using Snell's law
            float sin2Theta_i = glm::max<float>(0.0f, 1.0f - Sqr(cosTheta_i));
            float sin2Theta_t = sin2Theta_i / Sqr(eta);
            // Handle total internal reflection case
            if (sin2Theta_t >= 1.0f)
                return false;

            float cosTheta_t = glm::sqrt(1.0f - sin2Theta_t);

            *wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * n;
            // Provide relative IOR along ray to caller
            if (etap)
                *etap = eta;

            return true;
        }



        __device__ __host__ glm::vec3 f(float roughness, glm::vec3 wo, glm::vec3 wi) {
            const float eta = 1.5f;
            const float alpha = Surface::GetAlpha(roughness);

            if (eta == 1 || Surface::IsEffectifvelySmooth(roughness))
                return glm::vec3(0.0f);

            // Evaluate rough dielectric BSDF
            // Compute generalized half vector _wm_
            float cosTheta_o = SpherGeom::CosTheta(wo);
            float cosTheta_i = SpherGeom::CosTheta(wi);

            bool reflect = cosTheta_i * cosTheta_o > 0;
            float etap = 1;
            if (!reflect)
                etap = cosTheta_o > 0.0f ? eta : (1.0f / eta);

            glm::vec3 wm = wi * etap + wo;
            //CHECK_RARE(1e-5f, LengthSquared(wm) == 0);
            if (cosTheta_i == 0.0f || cosTheta_o == 0.0f || Sqr(glm::length(wm)) == 0.0f)
                return glm::vec3(0.0f);
            //wm = FaceForward(Normalize(wm), Normal3f(0, 0, 1));
            wm = glm::faceforward(-glm::normalize(wm), glm::vec3(0, 0, 1), glm::normalize(wm));

            // Discard backfacing microfacets
            if (glm::dot(wm, wi) * cosTheta_i < 0.0f || glm::dot(wm, wo) * cosTheta_o < 0)
                return glm::vec3(0.0f);

            float F = FresnelDielectric(glm::dot(wo, wm), eta);
            if (reflect) {
                // Compute reflection at rough dielectric interface
                return glm::vec3(Microfacet::D_Isotropic(wm, alpha) * Microfacet::G_Isotropic(wo, wi, alpha) * F / std::abs(4.0f * cosTheta_i * cosTheta_o));
            }
            else {
                // Compute transmission at rough dielectric interface
                float denom = Sqr(glm::dot(wi, wm) + glm::dot(wo, wm) / etap) * cosTheta_i * cosTheta_o;
                float ft = Microfacet::D_Isotropic(wm, alpha) * (1.0f - F) * Microfacet::G_Isotropic(wo, wi, alpha) * std::abs(glm::dot(wi, wm) * glm::dot(wo, wm) / denom);
                // Account for non-symmetry with transmission to different medium
                //if (mode == TransportMode::Radiance)
                ft /= Sqr(etap);

                return glm::vec3(ft);
            }
        }





        //__device__ __host__ bool Sample_f(unsigned int& randomSeed, float roughness, glm::vec3 wo, BSDFSample& sample)
        __device__ __host__ bool Sample_f(unsigned int& randomSeed, float roughness, glm::vec3 wo, BSDFSample& sample, bool reflection = true, bool transmission = true) {
            const float eta = 1.5f;
            const float alpha = Surface::GetAlpha(roughness);
            const float uc = RandomOptix::rnd(randomSeed);

            if (eta == 1 || Surface::IsEffectifvelySmooth(roughness)) {
                // Sample perfect specular dielectric BSDF
                float R = FresnelDielectric(SpherGeom::CosTheta(wo), eta), T = 1 - R;

                // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
                float pr = R, pt = T;
                if (!reflection)//!(sampleFlags & BxDFReflTransFlags::Reflection)
                    pr = 0.0f;
                if (!transmission)//!(sampleFlags & BxDFReflTransFlags::Transmission)
                    pt = 0.0f;
                if (pr == 0.0f && pt == 0.0f)
                    return false;

                if (uc < pr / (pr + pt)) {
                    // Sample perfect specular dielectric BRDF
                    glm::vec3 wi(-wo.x, -wo.y, wo.z);
                    glm::vec3 fr(R / SpherGeom::AbsCosTheta(wi));

                    sample.color = fr;
                    sample.direction = wi;
                    sample.pdf = pr / (pr + pt);

                    sample.reflection = true;
                    sample.transmission = false;
                    sample.glossy = false;
                    sample.specular = true;

                    return true;
                    //return BSDFSample(fr, wi, pr / (pr + pt), BxDFFlags::SpecularReflection);

                }
                else {

                    // Sample perfect specular dielectric BTDF
                    // Compute ray direction for specular transmission
                    glm::vec3 wi;
                    float etap;

                    bool valid = Refract(wo, glm::vec3(0, 0, 1), eta, &etap, &wi);

                    //CHECK_RARE(1e-5f, !valid);
                    if (!valid)
                        return false;

                    glm::vec3 ft(T / SpherGeom::AbsCosTheta(wi));
                    // Account for non-symmetry with transmission to different medium
                    //if (mode == TransportMode::Radiance)
                    ft /= Sqr(etap);

                    sample.color = ft;
                    sample.direction = wi;
                    sample.pdf = pt / (pr + pt);

                    sample.reflection = false;
                    sample.transmission = true;
                    sample.glossy = false;
                    sample.specular = true;

                    return true;
                    //return BSDFSample(ft, wi, pt / (pr + pt), BxDFFlags::SpecularTransmission, etap);
                }

            }
            else {
                // Sample rough dielectric BSDF
                glm::vec3 wm = Microfacet::Sample_wm(randomSeed, wo, alpha);
                float R = FresnelDielectric(glm::dot(wo, wm), eta);
                float T = 1.0f - R;
                // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
                float pr = R, pt = T;
                if (!reflection)//!(sampleFlags & BxDFReflTransFlags::Reflection)
                    pr = 0.0f;
                if (!transmission)//!(sampleFlags & BxDFReflTransFlags::Transmission)
                    pt = 0.0f;
                if (pr == 0.0f && pt == 0.0f)
                    return false;

                float pdf;
                if (uc < pr / (pr + pt)) {
                    // Sample reflection at rough dielectric interface
                    glm::vec3 wi = glm::reflect(-wo, wm);
                    if (!SpherGeom::SameHemisphere(wo, wi))
                        return false;
                    // Compute PDF of rough dielectric reflection
                    pdf = Microfacet::PDF_Isotropic(wo, wm, alpha) / (4 * AbsDot(wo, wm)) * pr / (pr + pt);

                    //DCHECK(!IsNaN(pdf));
                    glm::vec3 f(Microfacet::D_Isotropic(wm, alpha) * Microfacet::G_Isotropic(wo, wi, alpha) * R / (4.0f * SpherGeom::CosTheta(wi) * SpherGeom::CosTheta(wo)));

                    sample.color = f;
                    sample.direction = wi;
                    sample.pdf = pdf;

                    sample.reflection = true;
                    sample.transmission = false;
                    sample.glossy = true;
                    sample.specular = false;

                    return true;
                    //return BSDFSample(f, wi, pdf, BxDFFlags::GlossyReflection);

                }
                else {
                    // Sample transmission at rough dielectric interface
                    float etap;
                    glm::vec3 wi;
                    bool tir = !Refract(wo, wm, eta, &etap, &wi);
                    //CHECK_RARE(1e-5f, tir);
                    if (SpherGeom::SameHemisphere(wo, wi) || wi.z == 0 || tir)
                        return false;
                    // Compute PDF of rough dielectric transmission
                    float denom = Sqr(glm::dot(wi, wm) + glm::dot(wo, wm) / etap);
                    float dwm_dwi = AbsDot(wi, wm) / denom;
                    pdf = Microfacet::PDF_Isotropic(wo, wm, alpha) * dwm_dwi * pt / (pr + pt);

                    //CHECK(!IsNaN(pdf));
                    // Evaluate BRDF and return _BSDFSample_ for rough transmission
                    glm::vec3 ft(T * Microfacet::D_Isotropic(wm, alpha) * Microfacet::G_Isotropic(wo, wi, alpha) * std::abs(glm::dot(wi, wm) * glm::dot(wo, wm) /
                            (SpherGeom::CosTheta(wi) * SpherGeom::CosTheta(wo) * denom)));
                    // Account for non-symmetry with transmission to different medium
                    //if (mode == TransportMode::Radiance)
                    ft /= Sqr(etap);

                    sample.color = ft;
                    sample.direction = wi;
                    sample.pdf = pdf;

                    sample.reflection = false;
                    sample.transmission = true;
                    sample.glossy = true;
                    sample.specular = false;

                    return true;
                    //return BSDFSample(ft, wi, pdf, BxDFFlags::GlossyTransmission, etap);
                }
            }
        }


        //__device__ __host__ float PDF(float roughness, glm::vec3 wo, glm::vec3 wi) {
        //    const float eta = 1.5f;

        //    //const float alpha = Sqr(roughness);
        //    const float alpha = sqrt(roughness);

        //    if (eta == 1 || Surface::IsEffectifvelySmooth(roughness))
        //        return 0;
        //    // Evaluate sampling PDF of rough dielectric BSDF
        //    // Compute generalized half vector _wm_
        //    float cosTheta_o = SpherGeom::CosTheta(wo);
        //    float cosTheta_i = SpherGeom::CosTheta(wi);

        //    bool reflect = cosTheta_i * cosTheta_o > 0;
        //    float etap = 1;
        //    if (!reflect)
        //        etap = cosTheta_o > 0 ? eta : (1 / eta);
        //    glm::vec3 wm = wi * etap + wo;
        //    //CHECK_RARE(1e-5f, LengthSquared(wm) == 0);

        //    if (cosTheta_i == 0 || cosTheta_o == 0 || Sqr(glm::length(wm)) == 0)
        //        return {};

        //    //wm = FaceForward(Normalize(wm), Normal3f(0, 0, 1));
        //    wm = glm::faceforward(-glm::normalize(wm), glm::vec3(0, 0, 1), glm::normalize(wm));

        //    // Discard backfacing microfacets
        //    if (glm::dot(wm, wi) * cosTheta_i < 0 || glm::dot(wm, wo) * cosTheta_o < 0)
        //        return {};

        //    // Determine Fresnel reflectance of rough dielectric boundary
        //    float R = FresnelDielectric(glm::dot(wo, wm), eta);
        //    float T = 1 - R;

        //    // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
        //    float pr = R, pt = T;
        //    /*if (!(sampleFlags & BxDFReflTransFlags::Reflection))
        //        pr = 0;
        //    if (!(sampleFlags & BxDFReflTransFlags::Transmission))
        //        pt = 0;*/
        //    if (pr == 0 && pt == 0)
        //        return {};

        //    // Return PDF for rough dielectric
        //    float pdf;
        //    if (reflect) {
        //        // Compute PDF of rough dielectric reflection
        //        pdf = Microfacet::PDF_Isotropic(wo, wm, alpha) / (4 * AbsDot(wo, wm)) * pr / (pr + pt);

        //    }
        //    else {
        //        // Compute PDF of rough dielectric transmission
        //        float denom = Sqr(glm::dot(wi, wm) + glm::dot(wo, wm) / etap);
        //        float dwm_dwi = AbsDot(wi, wm) / denom;
        //        pdf = Microfacet::PDF_Isotropic(wo, wm, alpha) * dwm_dwi * pt / (pr + pt);
        //    }
        //    return pdf;
        //}
    }
}