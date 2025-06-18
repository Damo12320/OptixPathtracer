#pragma once

#include "../random.h"
#include "SphericalGeometry.h"

namespace PBRT { namespace Microfacet {

#pragma region MicrofacetDistribution

    __device__ __host__ float D_Anisotropic(glm::vec3 wm, glm::vec2 alpha) {
        float tan2Theta = SpherGeom::Tan2Theta(wm);
        if (isinf(tan2Theta)) return 0;

        float cos4Theta = Sqr(SpherGeom::Cos2Theta(wm));
        float e = tan2Theta * (Sqr(SpherGeom::CosPhi(wm) / alpha.x) +
            Sqr(SpherGeom::SinPhi(wm) / alpha.y));

        const float pi = 3.14159265359;
        return 1 / (pi * alpha.x * alpha.y * cos4Theta * Sqr(1 + e));
    }

    __device__ __host__ float D_Isotropic(glm::vec3 wm, float alpha) {
        // Gather main ingredients
        const float cos2Theta = SpherGeom::Cos2Theta(wm);
        const float tan2Theta = SpherGeom::Tan2Theta(wm);

        float alpha2 = Sqr(alpha);

        // The actual Distribution Term
        float mainTerm = 1 + (tan2Theta / alpha2);
        mainTerm = mainTerm * mainTerm;

        // The normaliing Term
        const float pi = 3.14159265359;
        float normalizing = pi * alpha2 * Sqr(cos2Theta);

        return 1 / (normalizing * mainTerm);
    }

#pragma endregion

#pragma region MaskingShadowingFunctions

    __device__ __host__ float Lambda_Isotropic(glm::vec3 w, float alpha) {
        float tan2Theta = SpherGeom::Tan2Theta(w);
        if (isinf(tan2Theta)) return 0;

        float alpha2 = Sqr(alpha);
        return (std::sqrt(1 + alpha2 * tan2Theta) - 1) / 2;
    }

    __device__ __host__ float G_Isotropic(glm::vec3 wo, glm::vec3 wi, float alpha) {
        return 1 / (1 + Lambda_Isotropic(wo, alpha) + Lambda_Isotropic(wi, alpha));
    }

    __device__ __host__ float G1_Isotropic(glm::vec3 w, float alpha) {
        return 1 / (1 + Lambda_Isotropic(w, alpha));
    }

#pragma endregion

    __device__ __host__ float D_Isotropic(glm::vec3 w, glm::vec3 wm, float alpha) {
        return G1_Isotropic(w, alpha) / SpherGeom::AbsCosTheta(w) * D_Isotropic(wm, alpha) * AbsDot(w, wm);
    }

    __device__ __host__ float PDF_Isotropic(glm::vec3 w, glm::vec3 wm, float alpha) {
        return D_Isotropic(w, wm, alpha);
    }

    __device__ __host__ glm::vec3 Sample_wm(unsigned int& randomSeed, glm::vec3 w, float alpha) {
        //<< Transform w to hemispherical configuration >>
        glm::vec3 wh = glm::normalize(glm::vec3(alpha * w.x, alpha * w.y, w.z));
        if (wh.z < 0.0f)
            wh = -wh;

        //<< Find orthonormal basis for visible normal sampling >>
        glm::vec3 T1 = (wh.z < 0.99999f) ? glm::normalize(glm::cross(glm::vec3(0, 0, 1), wh)) : glm::vec3(1, 0, 0);
        glm::vec3 T2 = glm::cross(wh, T1);

        //<< Generate uniformly distributed points on the unit disk >>
        glm::vec2 p = RandomOptix::SampleUniformDiskPolar(randomSeed);

        //<< Warp hemispherical projection for visible normal sampling >>
        float h = sqrtf(1 - Sqr(p.x));
        p.y = glm::mix(h, p.y, (1.0f + wh.z) / 2.0f);

        //<< Reproject to hemisphere and transform normal to ellipsoid configuration >>
        float pz = sqrtf(glm::max(0.0f, 1.0f - Sqr(glm::length(p))));
        glm::vec3 nh = p.x * T1 + p.y * T2 + pz * wh;
        return glm::normalize(glm::vec3(alpha * nh.x, alpha * nh.y, glm::max(1e-6f, nh.z)));
    }
} }