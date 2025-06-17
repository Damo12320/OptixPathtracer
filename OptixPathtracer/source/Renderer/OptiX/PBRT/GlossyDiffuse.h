#pragma once

#include "Dielectric.h"
#include "LambertDiffuse.h"

namespace PBRT {
	namespace GlossyDiffuse {

#pragma region PhaseFunctions
        struct PhaseFunctionSample {
            float p;
            glm::vec3 wi;
            float pdf;
        };

        struct Frame {
            glm::vec3 x, y, z;

            __device__ Frame(glm::vec3 x, glm::vec3 y, glm::vec3 z) {
                this->x = x;
                this->y = y;
                this->z = z;
            }

            __device__ glm::vec3 FromLocal(glm::vec3 v) const {
                return v.x * x + v.y * y + v.z * z;
            }
        };

        __device__ void CoordinateSystem(glm::vec3 v1, glm::vec3* v2, glm::vec3* v3) {
            float sign = copysign(float(1), v1.z);
            float a = -1 / (sign + v1.z);
            float b = v1.x * v1.y * a;
            *v2 = glm::vec3(1 + sign * Sqr(v1.x) * a, sign * b, -sign * v1.x);
            *v3 = glm::vec3(b, sign + Sqr(v1.y) * a, -v1.y);
        }

        __device__ Frame FromZ(glm::vec3 z) {
            glm::vec3 x, y;
            CoordinateSystem(z, &x, &y);
            return Frame(x, y, z);
        }

        __device__ float HenyeyGreenstein(float cosTheta, float g) {
            const float inv4Pi = 0.07957747154594766788f;

            float denom = 1 + Sqr(g) + 2 * g * cosTheta;
            return inv4Pi * (1 - Sqr(g)) / (denom * sqrt(denom));
        }

        __device__ glm::vec3 SampleHenyeyGreenstein(glm::vec3 wo, float g, glm::vec2 u, float* pdf) {
            const float pi = 3.14159265358979323846;

            //<< Compute for Henyey–Greenstein sample >>
            float cosTheta;
            if (std::abs(g) < 1e-3f)
                cosTheta = 1 - 2 * u[0];
            else
                cosTheta = -1 / (2 * g) *
                (1 + Sqr(g) - Sqr((1 - Sqr(g)) / (1 + g - 2 * g * u[0])));

            //<< Compute direction wi for Henyey–Greenstein sample >>
            float sinTheta = sqrt(1 - Sqr(cosTheta));
            float phi = 2 * pi * u[1];
            Frame wFrame = FromZ(wo);
            glm::vec3 wi = wFrame.FromLocal(SpherGeom::SphericalDirection(sinTheta, cosTheta, phi));
            //glm::vec3 wi = SpherGeom::SphericalDirection(sinTheta, cosTheta, phi);

            if (pdf) *pdf = HenyeyGreenstein(cosTheta, g);
            return wi;
        }

        __device__ PhaseFunctionSample PhaseFunction_Sample_p(glm::vec3 wo, glm::vec2 u, float g) {
            float pdf;
            glm::vec3 wi = SampleHenyeyGreenstein(wo, g, u, &pdf);
            return PhaseFunctionSample{ pdf, wi, pdf };
        }

        __device__ float PhaseFunction_p(glm::vec3 wo, glm::vec3 wi, float g) {
            return HenyeyGreenstein(glm::dot(wo, wi), g);
        }

        __device__ float PDF(glm::vec3 wo, glm::vec3 wi, float g) {
            return PhaseFunction_p(wo, wi, g);
        }

#pragma endregion

        __device__ float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
            float f = nf * fPdf, g = ng * gPdf;
            return Sqr(f) / (Sqr(f) + Sqr(g));
        }

        __device__ float Transmittance(float dz, glm::vec3 w) {
            //return FastExp(-std::abs(dz / w.z));
            return glm::exp(-std::abs(dz / w.z));
        }















        // Top interface: Dielectric
        // Bottom interface: Diffuse
        // Always enter on top

        __device__ glm::vec3 f(unsigned int& randomSeed, Surface& surface, glm::vec3 wi) {
            const int nSamples = 5;
            const float thickness = 0.0001f;
            const int maxDepth = 5;

            //scattering of light (0 = isotropic)
            const float g = 0;

            glm::vec3 wo = surface.outgoingRay;

            glm::vec3 f(0.);
            // Estimate _LayeredBxDF_ value _f_ using random sampling
            // Set _wo_ and _wi_ for layered BSDF evaluation
            /*if (twoSided && wo.z < 0) {
                wo = -wo;
                wi = -wi;
            }*/ //ONLY ONE SIDED

            // Determine entrance interface for layered BSDF
            /*TopOrBottomBxDF<TopBxDF, BottomBxDF> enterInterface;
            bool enteredTop = twoSided || wo.z > 0;
            if (enteredTop)
                enterInterface = &top;
            else
                enterInterface = &bottom;*/

            // Determine exit interface and exit $z$ for layered BSDF
            /*TopOrBottomBxDF<TopBxDF, BottomBxDF> exitInterface, nonExitInterface;
            if (SameHemisphere(wo, wi) ^ enteredTop) {
                exitInterface = &bottom;
                nonExitInterface = &top;
            }
            else {
                exitInterface = &top;
                nonExitInterface = &bottom;
            }*/
            float exitZ = (SpherGeom::SameHemisphere(wo, wi) ^ true) ? 0 : thickness;

            // Account for reflection at the entrance interface
            if (SpherGeom::SameHemisphere(wo, wi))
                f = (float)nSamples * Dielectric::f(surface.roughness, surface.outgoingRay, wi);
                //f = nSamples * enterInterface.f(wo, wi, mode);

            // Declare _RNG_ for layered BSDF evaluation
            /*RNG rng(Hash(GetOptions().seed, wo), Hash(wi));
            auto r = [&rng]() {
                return std::min<float>(rng.Uniform<float>(), OneMinusEpsilon);
                };*/

            auto r = [&randomSeed]() {
                return RandomOptix::rnd(randomSeed);
                };

            for (int s = 0; s < nSamples; ++s) {
                // Sample random walk through layers to estimate BSDF value
                // Sample transmission direction through entrance interface
                //float uc = r();

                BSDFSample wos;
                bool wos_test = Dielectric::Sample_f(randomSeed, surface.roughness, surface.outgoingRay, wos);
                if (!wos_test || wos.color == glm::vec3(0) || wos.pdf == 0 || wos.direction.z == 0)
                    continue;

                /*pstd::optional<BSDFSample> wos = enterInterface.Sample_f(
                    wo, uc, Point2f(r(), r()), mode, BxDFReflTransFlags::Transmission);
                if (!wos || !wos->f || wos->pdf == 0 || wos->wi.z == 0)
                    continue;*/

                // Sample BSDF for virtual light from _wi_

                BSDFSample wis;
                bool wis_test = LambertDiffuse::Sample_f(randomSeed, surface.albedo, surface.roughness, surface.outgoingRay, wis);
                if (!wis_test || wis.color == glm::vec3(0) || wis.pdf == 0 || wis.direction.z == 0)
                    continue;

                /*uc = r();
                pstd::optional<BSDFSample> wis = exitInterface.Sample_f(
                    wi, uc, Point2f(r(), r()), !mode, BxDFReflTransFlags::Transmission);
                if (!wis || !wis->f || wis->pdf == 0 || wis->wi.z == 0)
                    continue;*/

                // Declare state for random walk through BSDF layers
                glm::vec3 beta = wos.color * SpherGeom::AbsCosTheta(wos.direction) / wos.pdf;
                float z = true ? thickness : 0;//enteredTop ?
                glm::vec3 w = wos.direction;
                //HGPhaseFunction phase(g);

                for (int depth = 0; depth < maxDepth; ++depth) {
                    // Sample next event for layered BSDF evaluation random walk
                    /*PBRT_DBG("beta: %f %f %f %f, w: %f %f %f, f: %f %f %f %f\n", beta[0],
                        beta[1], beta[2], beta[3], w.x, w.y, w.z, f[0], f[1], f[2],
                        f[3]);*/
                    // Possibly terminate layered BSDF random walk with Russian roulette
                    if (depth > 3 && SaveMax(beta) < 0.25f) {
                        float q = glm::max(0.0f, 1.0f - SaveMax(beta));
                        if (r() < q)
                            break;
                        beta /= 1 - q;
                        /*PBRT_DBG("After RR with q = %f, beta: %f %f %f %f\n", q, beta[0],
                            beta[1], beta[2], beta[3]);*/
                    }

                    // Account for media between layers and possibly scatter
                    if (surface.albedo == glm::vec3(0)) {
                        // Advance to next layer boundary and update _beta_ for transmittance
                        z = (z == thickness) ? 0 : thickness;
                        beta *= Transmittance(thickness, w);

                    }
                    else {
                        // Sample medium scattering for layered BSDF evaluation
                        float sigma_t = 1;
                        //float dz = SampleExponential(r(), sigma_t / std::abs(w.z));
                        float dz = -glm::log(1 - r()) / (sigma_t / std::abs(w.z));
                        float zp = w.z > 0 ? (z + dz) : (z - dz);
                        //DCHECK_RARE(1e-5, z == zp);
                        if (z == zp)
                            continue;

                        if (0 < zp && zp < thickness) {
                            // Handle scattering event in layered BSDF medium
                            // Account for scattering through _exitInterface_ using _wis_
                            float wt = 1;
                            //if (!IsSpecular(exitInterface.Flags()))
                                //wt = PowerHeuristic(1, wis->pdf, 1, phase.PDF(-w, -wis->wi));
                            f += beta * surface.albedo * PhaseFunction_p(-w, -wis.direction, g) * wt *
                                Transmittance(zp - exitZ, wis.direction) * wis.color / wis.pdf;

                            // Sample phase function and update layered path state
                            glm::vec2 u{ r(), r() };
                            PhaseFunctionSample ps = PhaseFunction_Sample_p(-w, u, g);
                            if (ps.pdf == 0 || ps.wi.z == 0)
                                continue;
                            beta *= surface.albedo * ps.p / ps.pdf;
                            w = ps.wi;
                            z = zp;

                            // Possibly account for scattering through _exitInterface_
                            if ((z < exitZ && w.z > 0) || (z > exitZ && w.z < 0)) {//&& !IsSpecular(exitInterface.Flags())
                                // Account for scattering through _exitInterface_
                                //glm::vec3 fExit = LambertDiffuse::f(-w, wi);
                                glm::vec3 fExit = LambertDiffuse::f(surface.roughness, surface.albedo, -w, wi);
                                if (fExit != glm::vec3(0)) {
                                    //float exitPDF = LambertDiffuse::PDF(-w, wi, mode, BxDFReflTransFlags::Transmission);
                                    float exitPDF = LambertDiffuse::PDF(-w, wi);
                                    float wt = PowerHeuristic(1, ps.pdf, 1, exitPDF);
                                    f += beta * Transmittance(zp - exitZ, ps.wi) * fExit * wt;
                                }
                            }

                            continue;
                        }
                        z = glm::clamp(zp, 0.0f, thickness);
                    }

                    // Account for scattering at appropriate interface
                    if (z == exitZ) {
                        // Account for reflection at _exitInterface_
                        //float uc = r();
                        //pstd::optional<BSDFSample> bs = LambertDiffuse::Sample_f( -w, uc, Point2f(r(), r()), mode, BxDFReflTransFlags::Reflection);
                        //pstd::optional<BSDFSample> bs = LambertDiffuse::Sample_f( -w, uc, Point2f(r(), r()), mode, BxDFReflTransFlags::Reflection);

                        BSDFSample bs;
                        bool bs_test = LambertDiffuse::Sample_f(randomSeed, surface.albedo, surface.roughness, -w, bs);
                        if (!bs_test || bs.color == glm::vec3(0) || bs.pdf == 0 || bs.direction.z == 0)
                            break;
                        beta *= bs.color * SpherGeom::AbsCosTheta(bs.direction) / bs.pdf;
                        w = bs.direction;

                    }
                    else {
                        // Account for scattering at _nonExitInterface_
                        if (!surface.IsEffectifvelySmooth()) {//!IsSpecular(nonExitInterface.Flags())
                            // Add NEE contribution along presampled _wis_ direction
                            float wt = 1;
                            //if (!IsSpecular(exitInterface.Flags()))
                                //wt = PowerHeuristic(1, wis->pdf, 1, nonExitInterface.PDF(-w, -wis->wi, mode));
                            wt = PowerHeuristic(1, wis.pdf, 1, Dielectric::PDF(surface.roughness, -w, -wis.direction));

                            f += beta * Dielectric::f(surface.roughness, -w, -wis.direction) *
                                SpherGeom::AbsCosTheta(wis.direction) * wt * Transmittance(thickness, wis.direction) * wis.color / wis.pdf;
                        }
                        // Sample new direction using BSDF at _nonExitInterface_
                        //float uc = r();
                        //Point2f u(r(), r());
                        //pstd::optional<BSDFSample> bs = nonExitInterface.Sample_f(-w, uc, u, mode, BxDFReflTransFlags::Reflection);

                        BSDFSample bs;
                        bool bs_test = Dielectric::Sample_f(randomSeed, surface.roughness, -w, bs);
                        if (!bs_test || bs.color == glm::vec3(0) || bs.pdf == 0 || bs.direction.z == 0)
                            break;
                        beta *= bs.color * SpherGeom::AbsCosTheta(bs.direction) / bs.pdf;
                        w = bs.direction;

                        if (true) {//!IsSpecular(exitInterface.Flags())
                            // Add NEE contribution along direction from BSDF sample
                            //glm::vec3 fExit = exitInterface.f(-w, wi, mode);
                            glm::vec3 fExit = LambertDiffuse::f(surface.roughness, surface.albedo, -w, wi);
                            if (fExit != glm::vec3(0)) {
                                float wt = 1;
                                if (!surface.IsEffectifvelySmooth()) {//!IsSpecular(nonExitInterface.Flags())
                                    //float exitPDF = exitInterface.PDF( -w, wi, mode, BxDFReflTransFlags::Transmission);
                                    float exitPDF = LambertDiffuse::PDF(-w, wi);
                                    wt = PowerHeuristic(1, bs.pdf, 1, exitPDF);
                                }
                                f += beta * Transmittance(thickness, bs.direction) * fExit * wt;
                            }
                        }
                    }
                }
            }

            return f / (float)nSamples;
        }















        __device__ bool Sample_f(unsigned int& randomSeed, Surface& surface, BSDFSample& sample) {
            const int nSamples = 5;
            const float thickness = 0.0001f;
            const int maxDepth = 5;

            //scattering of light (0 = isotropic)
            const float g = 0;


            //CHECK(sampleFlags == BxDFReflTransFlags::All);  // for now
            // Set _wo_ for layered BSDF sampling
            bool flipWi = false;
            /*if (twoSided&& wo.z < 0) {
                wo = -wo;
                flipWi = true;
            }*/

            // Sample BSDF at entrance interface to get initial direction _w_
            //bool enteredTop = twoSided || wo.z > 0;
            bool enteredTop = true;
            //pstd::optional<BSDFSample> bs = enteredTop ? top.Sample_f(wo, uc, u, mode) : bottom.Sample_f(wo, uc, u, mode);
            //pstd::optional<BSDFSample> bs = enteredTop ? top.Sample_f(wo, uc, u, mode) : bottom.Sample_f(wo, uc, u, mode);
            BSDFSample bs;
            bool bs_test = Dielectric::Sample_f(randomSeed, surface.roughness, surface.outgoingRay, bs);
            if (!bs_test || bs.color == glm::vec3(0) || bs.pdf == 0 || bs.direction.z == 0)
                return false;

            if (bs.reflection) {//bs->IsReflection()
                if (flipWi)
                    bs.direction = -bs.direction;
                //bs->pdfIsProportional = true;
                sample = bs;
                return true;
            }
            glm::vec3 w = bs.direction;
            //bool specularPath = bs->IsSpecular();
            bool specularPath = true;

            // Declare _RNG_ for layered BSDF sampling
            /*RNG rng(Hash(GetOptions().seed, wo), Hash(uc, u));
            auto r = [&rng]() {
                return std::min<float>(rng.Uniform<float>(), OneMinusEpsilon);
                };*/
            auto r = [&randomSeed]() {
                return RandomOptix::rnd(randomSeed);
                };

            // Declare common variables for layered BSDF sampling
            glm::vec3 f = bs.color * SpherGeom::AbsCosTheta(bs.direction);
            float pdf = bs.pdf;
            //float z = enteredTop ? thickness : 0;
            float z = thickness;
            //HGPhaseFunction phase(g);

            for (int depth = 0; depth < maxDepth; ++depth) {
                // Follow random walk through layers to sample layered BSDF
                // Possibly terminate layered BSDF sampling with Russian Roulette
                float rrBeta = SaveMax(f) / pdf;
                if (depth > 3 && rrBeta < 0.25f) {
                    float q = glm::max(0.0f, 1.0f - rrBeta);
                    if (r() < q)
                        return {};
                    pdf *= 1 - q;
                }
                if (w.z == 0) return false;

                if (surface.albedo != glm::vec3(0)) {
                    // Sample potential scattering event in layered medium
                    float sigma_t = 1;
                    //float dz = SampleExponential(r(), sigma_t / AbsCosTheta(w));
                    float dz = -glm::log(1 - r()) / (sigma_t / SpherGeom::AbsCosTheta(w));
                    float zp = w.z > 0 ? (z + dz) : (z - dz);
                    //CHECK_RARE(1e-5, zp == z);
                    if (zp == z)
                        return {};
                    if (0 < zp && zp < thickness) {
                        // Update path state for valid scattering event between interfaces
                        PhaseFunctionSample ps = PhaseFunction_Sample_p(-w, glm::vec2(r(), r()), g);
                        if (ps.pdf == 0 || ps.wi.z == 0)
                            return false;
                        f *= surface.albedo * ps.p;
                        pdf *= ps.pdf;
                        specularPath = false;
                        w = ps.wi;
                        z = zp;

                        continue;
                    }
                    z = glm::clamp(zp, 0.0f, thickness);
                    //if (z == 0)
                        //DCHECK_LT(w.z, 0);
                    //else
                        //DCHECK_GT(w.z, 0);

                }
                else {
                    // Advance to the other layer interface
                    z = (z == thickness) ? 0 : thickness;
                    f *= Transmittance(thickness, w);
                }
                // Initialize _interface_ for current interface surface
//#ifdef interface  // That's enough out of you, Windows.
//#undef interface
//#endif
                //TopOrBottomBxDF<TopBxDF, BottomBxDF> interface;
                //if (z == 0)
                    //interface = &bottom;
                //else
                    //interface = &top;

                BSDFSample bs;
                bool bs_test;


                if (z == 0) {
                    //interface = &bottom;
                    bs_test = LambertDiffuse::Sample_f(randomSeed, surface.albedo, surface.roughness, surface.outgoingRay, bs);

                    specularPath &= false;
                }
                else {
                    //interface = &top;
                    bs_test = Dielectric::Sample_f(randomSeed, surface.roughness, surface.outgoingRay, bs);
                    specularPath &= true;
                }

                if (!bs_test || bs.color == glm::vec3(0) || bs.pdf == 0 || bs.direction.z == 0)
                    return false;

                // Sample interface BSDF to determine new path direction
                //float uc = r();
                //Point2f u(r(), r());
                //pstd::optional<BSDFSample> bs = interface.Sample_f(-w, uc, u, mode);
                //if (!bs || !bs->f || bs->pdf == 0 || bs->wi.z == 0)
                    //return {};
                f *= bs.color;
                pdf *= bs.pdf;
                //specularPath &= bs->IsSpecular();
                w = bs.direction;

                // Return _BSDFSample_ if path has left the layers
                if (bs.transmission) {
                    bool reflection = SpherGeom::SameHemisphere(surface.outgoingRay, w);
                    bool specular = specularPath;

                    //flags |= specularPath ? BxDFFlags::Specular : BxDFFlags::Glossy;
                    if (flipWi)
                        w = -w;

                    sample.color = f;
                    sample.direction = w;
                    sample.pdf = pdf;

                    sample.reflection = reflection;
                    sample.transmission = !reflection;
                    sample.specular = specular;
                    sample.glossy = !specular;

                    //return BSDFSample(f, w, pdf, flags, 1.f, true);
                    return true;
                }

                // Scale _f_ by cosine term after scattering at the interface
                f *= SpherGeom::AbsCosTheta(bs.direction);
            }
            return false;
        }
	}
}