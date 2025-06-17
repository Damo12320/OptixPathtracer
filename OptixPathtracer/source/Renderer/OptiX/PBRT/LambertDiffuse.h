#pragma once

#include "Microfacet.h"
#include "../Surface.h"
#include "../RayData.h"
#include "BSDFSample.h"

namespace PBRT {
	namespace LambertDiffuse {
        __device__  glm::vec3 RandomSphereDirection(unsigned int& seed) {
            //glm::vec2 h = Hash2(seed) * glm::vec2(2.0, 6.28318530718) - glm::vec2(1, 0);
            glm::vec2 h = glm::vec2(RandomOptix::rnd(seed), RandomOptix::rnd(seed));
            float phi = h.y;

            float temp = sqrtf(1.0f - h.x * h.x);
            float sinPhi = sinf(phi);
            float cosPhi = cosf(phi);

            return glm::vec3(temp * sinPhi, temp * cosPhi, h.x);
        }

        __device__ glm::vec3 RandomHemisphereDirection(unsigned int& seed, const glm::vec3 n) {
            glm::vec3 dir = glm::normalize(RandomSphereDirection(seed));

            // Wenn die Richtung unterhalb der Fläche liegt, spiegle sie
            if (glm::dot(dir, n) < 0.0f) {
                dir = -dir;
            }
            return dir;
        }


        __device__ glm::vec2 SampleUniformDiskConcentric(unsigned int& seed) {
            const float PiOver4 = 0.78539816339744830961f;
            const float PiOver2 = 1.57079632679489661923f;

            //<< Map u to and handle degeneracy at the origin >>
            glm::vec2 uOffset = 2.0f * glm::vec2(RandomOptix::rnd(seed), RandomOptix::rnd(seed)) - glm::vec2(1, 1);
            if (uOffset.x == 0.0f && uOffset.y == 0.0f)
                return glm::vec2(0.0f, 0.0f);

            //<< Apply concentric mapping to point >>
            float theta, r;
            if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
                r = uOffset.x;
                theta = PiOver4 * (uOffset.y / uOffset.x);
            }
            else {
                r = uOffset.y;
                theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
            }
            return r * glm::vec2(std::cos(theta), std::sin(theta));
        }

        __device__ glm::vec3 SampleCosineHemisphere(unsigned int& seed) {
            glm::vec2 d = SampleUniformDiskConcentric(seed);
            float z = sqrt(1 - Sqr(d.x) - Sqr(d.y));
            return glm::vec3(d.x, d.y, z);
        }

        __device__ float CosineHemispherePDF(float cosTheta) {
            const float invPi = 0.31830988618379067154f;
            return cosTheta * invPi;
        }





        __device__ glm::vec3 f(float roughness, glm::vec3 surfaceColor, glm::vec3 wo, glm::vec3 wi) {
            if (!SpherGeom::SameHemisphere(wo, wi))
                return glm::vec3(0.0f);

            return surfaceColor / 3.14159265359f;
        }

        //__device__ bool Sample_f(unsigned int& randomSeed, glm::vec3 surfaceColor, float roughness, glm::vec3 wo, BSDFSample& sample) {
        //    sample.pdf = 1 / (2 * 3.141592654f);//https://ameye.dev/notes/sampling-the-hemisphere/
        //    sample.color = surfaceColor / 3.14159265359f;
        //    sample.direction = RandomHemisphereDirection(randomSeed, glm::vec3(0, 0, 1));

        //    if (sample.direction.z < 0) {
        //        sample.direction.z *= -1.0f;
        //    }

        //    sample.reflection = true;
        //    sample.transmission = false;
        //    sample.glossy = false;
        //    sample.specular = false;
        //    return true;
        //}

        __device__ bool Sample_f(unsigned int& randomSeed, glm::vec3 surfaceColor, float roughness, glm::vec3 wo, BSDFSample& sample) {
            const float invPi = 0.31830988618379067154f;

            sample.direction = SampleCosineHemisphere(randomSeed);
            if (sample.direction.z < 00.0f) {
                sample.direction.z *= -1.0f;
            }

            sample.pdf = CosineHemispherePDF(SpherGeom::AbsCosTheta(sample.direction));
            sample.color = surfaceColor * invPi;

            sample.reflection = true;
            sample.transmission = false;
            sample.glossy = false;
            sample.specular = false;

            return true;
        }

        __device__ float PDF(glm::vec3 wo, glm::vec3 wi) {
            //return 1 / (2 * 3.141592654f);

            if (!SpherGeom::SameHemisphere(wi, wo)) return 0.0f;

            return CosineHemispherePDF(SpherGeom::AbsCosTheta(wi));
        }
	}
}