#pragma once

#include "../random.h"
#include "SphericalGeometry.h"

namespace PBRT { namespace Microfacet {

#pragma region MicrofacetDistribution
    __device__ __host__ float D_Anisotropic(glm::vec3 wm, glm::vec2 alpha) {
        const float pi = 3.14159265359f;

        float tan2Theta = SpherGeom::Tan2Theta(wm);
        if (isinf(tan2Theta))
            return 0.0f;
        float cos4Theta = Sqr(SpherGeom::Cos2Theta(wm));
        if (cos4Theta < 1e-16f)
            return 0.0f;
        float e = tan2Theta * (Sqr(SpherGeom::CosPhi(wm) / alpha.x) + Sqr(SpherGeom::SinPhi(wm) / alpha.y));
        return 1.0f / (pi * alpha.x * alpha.y * cos4Theta * Sqr(1.0f + e));
    }

    __device__ __host__ float D_Isotropic(glm::vec3 wm, float alpha) {
        return D_Anisotropic(wm, glm::vec2(alpha));

        //// Gather main ingredients
        //const float cos2Theta = SpherGeom::Cos2Theta(wm);
        //const float tan2Theta = SpherGeom::Tan2Theta(wm);

        //float alpha2 = Sqr(alpha);

        //// The actual Distribution Term
        //float mainTerm = 1.0f + (tan2Theta / alpha2);
        //mainTerm = mainTerm * mainTerm;

        //// The normaliing Term
        //const float pi = 3.14159265359f;
        //float normalizing = pi * alpha2 * Sqr(cos2Theta);

        //return 1 / (normalizing * mainTerm);
    }

#pragma endregion

#pragma region MaskingShadowingFunctions

    __device__ __host__ float Lambda_Anisotropic(glm::vec3 w, glm::vec2 alpha) {
        float tan2Theta = SpherGeom::Tan2Theta(w);
        if (isinf(tan2Theta))
            return 0.0f;
        float alpha2 = Sqr(SpherGeom::CosPhi(w) * alpha.x) + Sqr(SpherGeom::SinPhi(w) * alpha.y);
        return (glm::sqrt(1.0f + alpha2 * tan2Theta) - 1.0f) / 2.0f;
    }

    __device__ __host__ float Lambda_Isotropic(glm::vec3 w, float alpha) {
        float tan2Theta = SpherGeom::Tan2Theta(w);
        if (isinf(tan2Theta)) return 0.0f;

        float alpha2 = Sqr(alpha);
        return (std::sqrt(1 + alpha2 * tan2Theta) - 1.0f) / 2.0f;
    }

    __device__ __host__ float G_Anisotropic(glm::vec3 wo, glm::vec3 wi, glm::vec2 alpha) {
        return 1 / (1 + Lambda_Anisotropic(wo, alpha) + Lambda_Anisotropic(wi, alpha));
    }

    __device__ __host__ float G_Isotropic(glm::vec3 wo, glm::vec3 wi, float alpha) {
        return G_Anisotropic(wo, wi, glm::vec2(alpha));
        //return 1 / (1 + Lambda_Isotropic(wo, alpha) + Lambda_Isotropic(wi, alpha));
    }

    __device__ __host__ float G1_Anisotropic(glm::vec3 w, glm::vec2 alpha) {
        return 1.0f / (1.0f + Lambda_Anisotropic(w, alpha));
    }

    __device__ __host__ float G1_Isotropic(glm::vec3 w, float alpha) {
        return 1.0f / (1.0f + Lambda_Isotropic(w, alpha));
    }

#pragma endregion

    __device__ __host__ float D_Isotropic(glm::vec3 w, glm::vec3 wm, float alpha) {
        //return G1_Isotropic(w, alpha) / SpherGeom::AbsCosTheta(w) * D_Isotropic(wm, alpha) * AbsDot(w, wm);
        return G1_Anisotropic(w, glm::vec2(alpha, alpha)) / SpherGeom::AbsCosTheta(w) * D_Anisotropic(wm, glm::vec2(alpha, alpha)) * AbsDot(w, wm);
    }

    __device__ __host__ float PDF_Isotropic(glm::vec3 w, glm::vec3 wm, float alpha) {
        return D_Isotropic(w, wm, alpha);
    }

    __device__ __host__ glm::vec3 Sample_wm(unsigned int& randomSeed, glm::vec3 w, glm::vec2 alpha) {
        // Transform _w_ to hemispherical configuration
        glm::vec3 wh = glm::normalize(glm::vec3(alpha.x * w.x, alpha.y * w.y, w.z));
        if (wh.z < 0)
            wh = -wh;

        // Find orthonormal basis for visible normal sampling
        glm::vec3 T1 = (wh.z < 0.99999f) ? glm::normalize(glm::cross(glm::vec3(0, 0, 1), wh))
            : glm::vec3(1, 0, 0);
        glm::vec3 T2 = glm::cross(wh, T1);

        // Generate uniformly distributed points on the unit disk
        glm::vec2 p = RandomOptix::SampleUniformDiskPolar(randomSeed);

        // Warp hemispherical projection for visible normal sampling
        float h = glm::sqrt(1.0f - Sqr(p.x));

        auto Lerp = [](float x, float a, float b) {
            return (1.0f - x) * a + x * b;
            };

        p.y = Lerp((1 + wh.z) / 2, h, p.y);
        //p.y = glm::mix(h, p.y, (1.0f + wh.z) / 2.0f);

        // Reproject to hemisphere and transform normal to ellipsoid configuration
        float pz = std::sqrt(glm::max(0.0f, 1.0f - LengthSqr(p)));
        glm::vec3 nh = p.x * T1 + p.y * T2 + pz * wh;
        //CHECK_RARE(1e-5f, nh.z == 0);
        return glm::normalize(glm::vec3(alpha.x * nh.x, alpha.y * nh.y, glm::max(1e-6f, nh.z)));
    }

    __device__ __host__ glm::vec3 Sample_wm(unsigned int& randomSeed, glm::vec3 w, float alpha) {
        return Sample_wm(randomSeed, w, glm::vec2(alpha));

        ////<< Transform w to hemispherical configuration >>
        //glm::vec3 wh = glm::normalize(glm::vec3(alpha * w.x, alpha * w.y, w.z));
        //if (wh.z < 0.0f)
        //    wh = -wh;

        ////<< Find orthonormal basis for visible normal sampling >>
        //glm::vec3 T1 = (wh.z < 0.99999f) ? glm::normalize(glm::cross(glm::vec3(0, 0, 1), wh)) : glm::vec3(1, 0, 0);
        //glm::vec3 T2 = glm::cross(wh, T1);

        ////<< Generate uniformly distributed points on the unit disk >>
        //glm::vec2 p = RandomOptix::SampleUniformDiskPolar(randomSeed);

        ////<< Warp hemispherical projection for visible normal sampling >>
        //float h = glm::sqrt(1.0f - Sqr(p.x));
        //p.y = glm::mix(h, p.y, (1.0f + wh.z) / 2.0f);

        ////<< Reproject to hemisphere and transform normal to ellipsoid configuration >>
        //float pz = glm::sqrt(glm::max(0.0f, 1.0f - LengthSqr(p)));
        //glm::vec3 nh = p.x * T1 + p.y * T2 + pz * wh;
        //return glm::normalize(glm::vec3(alpha * nh.x, alpha * nh.y, glm::max(1e-6f, nh.z)));
    }
} }