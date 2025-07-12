#pragma once

#include "Dielectric.h"
#include "LambertDiffuse.h"

#include <cuda/std/limits>

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

            __device__ __host__ Frame(glm::vec3 x, glm::vec3 y, glm::vec3 z) {
                this->x = x;
                this->y = y;
                this->z = z;
            }

            __device__ __host__ glm::vec3 FromLocal(glm::vec3 v) const {
                return v.x * x + v.y * y + v.z * z;
            }
        };

        __device__ __host__ void CoordinateSystem(glm::vec3 v1, glm::vec3* v2, glm::vec3* v3) {
            float sign = copysign(float(1), v1.z);
            float a = -1.0f / (sign + v1.z);
            float b = v1.x * v1.y * a;
            *v2 = glm::vec3(1.0f + sign * Sqr(v1.x) * a, sign * b, -sign * v1.x);
            *v3 = glm::vec3(b, sign + Sqr(v1.y) * a, -v1.y);
        }

        __device__ __host__ Frame FromZ(glm::vec3 z) {
            glm::vec3 x, y;
            CoordinateSystem(z, &x, &y);
            return Frame(x, y, z);
        }

        __device__ __host__ float HenyeyGreenstein(float cosTheta, float g) {
            const float inv4Pi = 0.07957747154594766788f;

            float denom = 1.0f + Sqr(g) + 2.0f * g * cosTheta;
            return inv4Pi * (1.0f - Sqr(g)) / (denom * glm::sqrt( glm::max(denom, 0.0f) ));
        }

        __device__ __host__ glm::vec3 SampleHenyeyGreenstein(glm::vec3 wo, float g, glm::vec2 u, float* pdf) {
            const float pi = 3.14159265358979323846f;

            //<< Compute for Henyey–Greenstein sample >>
            float cosTheta;
            if (glm::abs(g) < 1e-3f)
                cosTheta = 1.0f - 2.0f * u[0];
            else
                cosTheta = -1.0f / (2.0f * g) *
                (1.0f + Sqr(g) - Sqr((1.0f - Sqr(g)) / (1.0f + g - 2.0f * g * u[0])));

            //<< Compute direction wi for Henyey–Greenstein sample >>
            float sinTheta = glm::sqrt( glm::max(1.0f - Sqr(cosTheta), 0.0f) );
            float phi = 2.0f * pi * u[1];
            Frame wFrame = FromZ(wo);
            glm::vec3 wi = wFrame.FromLocal(SpherGeom::SphericalDirection(sinTheta, cosTheta, phi));//<--------------THIS COULD BE THE PROBLEM! RIGHT VS LEFT HANDED COORDINATES (maybe)     THIS CANT BE THE PROBLEM WHEN MEDIAALBEDO == 0
            //glm::vec3 wi = SpherGeom::SphericalDirection(sinTheta, cosTheta, phi);

            if (pdf) *pdf = HenyeyGreenstein(cosTheta, g);
            return wi;
        }

        __device__ __host__ PhaseFunctionSample PhaseFunction_Sample_p(glm::vec3 wo, glm::vec2 u, float g) {
            float pdf;
            glm::vec3 wi = SampleHenyeyGreenstein(wo, g, u, &pdf);
            return PhaseFunctionSample{ pdf, wi, pdf };
        }

        __device__ __host__ float PhaseFunction_p(glm::vec3 wo, glm::vec3 wi, float g) {
            return HenyeyGreenstein(glm::dot(wo, wi), g);
        }

        __device__ __host__ float Phase_PDF(glm::vec3 wo, glm::vec3 wi, float g) {
            return PhaseFunction_p(wo, wi, g);
        }

#pragma endregion

        __device__ __host__ float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
            float f = nf * fPdf;
            float g = ng * gPdf;
            return Sqr(f) / (Sqr(f) + Sqr(g));
        }

        __device__ float Transmittance(float dz, glm::vec3 w) {
            if (glm::abs(dz) <= ::cuda::std::numeric_limits<float>::min()) {
                return 1.0f;
            }

            //return FastExp(-glm::abs(dz / w.z));
            return glm::exp(-glm::abs(dz / w.z));
            //return __expf(-glm::abs(dz / w.z));//FastExp from cuda
        }




        __device__ glm::vec3 Layer_f(bool topLayer, Surface surface, glm::vec3 wo, glm::vec3 wi, Dielectric::TransportMode mode) {
            if (topLayer) {
                return Dielectric::f(surface, wo, wi, mode);
            }
            else {
                return LambertDiffuse::f(surface, wo, wi);
            }
        }

        __device__ bool Layer_Sample_f(bool topLayer, unsigned int& randomSeed, Surface surface, glm::vec3 wo, BSDFSample& sample, Dielectric::TransportMode mode, bool reflection, bool transmission) {
            if (topLayer) {
                return Dielectric::Sample_f(randomSeed, surface, wo, sample, mode, reflection, transmission);
            }
            else {
                return LambertDiffuse::Sample_f(randomSeed, surface, wo, sample, reflection);
            }
        }

        __device__ float Layer_PDF(bool topLayer, Surface surface, glm::vec3 wo, glm::vec3 wi, bool reflection, bool transmission) {
            if (topLayer) {
                return Dielectric::PDF(surface, wo, wi, reflection, transmission);
            }
            else {
                return LambertDiffuse::PDF(surface, wo, wi, reflection);
            }
        }


        // Top interface: Dielectric
        // Bottom interface: Diffuse
        // Two sided
        __device__ __host__ glm::vec3 f(unsigned int& randomSeed, Surface& surface, glm::vec3 wi) {
            Dielectric::TransportMode mode = Dielectric::TransportMode::Radiance;

            const int nSamples = 5;
            const float thickness = 0.01f;
            const int maxDepth = 10;

            const glm::vec3 mediaAlbedo = glm::vec3(0);//Color of the participating medium between the two layers

            //scattering of light (0 = isotropic)
            const float g = 0;

            bool twoSided = true;
            glm::vec3 wo = surface.outgoingRay;

            bool topLayer_isSpecular = surface.IsEffectifvelySmooth();
            bool bottomLayer_isSpecular = false;

            glm::vec3 f(0.0f);
            // Estimate _LayeredBxDF_ value _f_ using random sampling
            // Set _wo_ and _wi_ for layered BSDF evaluation
            if (twoSided&& wo.z < 0) {
                wo = -wo;
                wi = -wi;
            }

            // Determine entrance interface for layered BSDF
            //TopOrBottomBxDF<TopBxDF, BottomBxDF> enterInterface;
            bool enterInterface_top;
            bool enteredTop = twoSided || wo.z > 0;
            enterInterface_top = enteredTop;
            //if (enteredTop) {
            //    //enterInterface = &top;
            //}
            //else {
            //    //enterInterface = &bottom;
            //}

            // Determine exit interface and exit $z$ for layered BSDF
            //TopOrBottomBxDF<TopBxDF, BottomBxDF> exitInterface, nonExitInterface;
            bool exitInterface_top, nonExitInterface_top;
            bool exit_isSpecular, nonExit_isSpecular;
            if (SpherGeom::SameHemisphere(wo, wi) ^ enteredTop) {
                //exitInterface = &bottom;
                //nonExitInterface = &top;

                exit_isSpecular =   bottomLayer_isSpecular;
                nonExit_isSpecular =    topLayer_isSpecular;

                exitInterface_top = false;
                nonExitInterface_top = true;
            }
            else {
                //exitInterface = &top;
                //nonExitInterface = &bottom;

                exit_isSpecular =   topLayer_isSpecular;
                nonExit_isSpecular =    bottomLayer_isSpecular;

                exitInterface_top = true;
                nonExitInterface_top = false;
            }
            float exitZ = (SpherGeom::SameHemisphere(wo, wi) ^ enteredTop) ? 0 : thickness;

            // Account for reflection at the entrance interface
            if (SpherGeom::SameHemisphere(wo, wi))
                f = glm::vec3(nSamples) * Layer_f(enterInterface_top, surface, wo, wi, mode);

            // Declare _RNG_ for layered BSDF evaluation
            /*RNG rng(Hash(GetOptions().seed, wo), Hash(wi));
            auto r = [&rng]() {
                return std::min<float>(rng.Uniform<float>(), OneMinusEpsilon);
                };*/

            unsigned int newSeed = RandomOptix::tea<16>(wo.x * 1000, wo.y * 1000);
            newSeed = RandomOptix::tea<16>(newSeed, wi.x * 1000);
            newSeed = RandomOptix::tea<16>(newSeed, wi.y * 1000);
            newSeed = RandomOptix::tea<16>(newSeed, randomSeed);

            auto r = [&newSeed]() {
                return RandomOptix::rnd(newSeed);
                };

            for (int s = 0; s < nSamples; ++s) {
                // Sample random walk through layers to estimate BSDF value
                // Sample transmission direction through entrance interface
                //float uc = r();
                //pstd::optional<BSDFSample> wos = enterInterface.Sample_f(wo, uc, Point2f(r(), r()), mode, BxDFReflTransFlags::Transmission);
                BSDFSample wos;
                bool wos_test = Layer_Sample_f(enterInterface_top, randomSeed, surface, wo, wos, mode, false, true);
                if (!wos_test || wos.color == glm::vec3(0) || wos.pdf == 0 || wos.direction.z == 0)
                    continue;

                // Sample BSDF for virtual light from _wi_
                //uc = r();
                //pstd::optional<BSDFSample> wis = exitInterface.Sample_f(wi, uc, Point2f(r(), r()), !mode, BxDFReflTransFlags::Transmission);
                BSDFSample wis;
                bool wis_test = Layer_Sample_f(exitInterface_top, randomSeed, surface, wi, wis, FlipMode(mode), false, true);
                if (!wis_test || wis.color == glm::vec3(0) || wis.pdf == 0 || wis.direction.z == 0)
                    continue;

                // Declare state for random walk through BSDF layers
                glm::vec3 beta = wos.color * SpherGeom::AbsCosTheta(wos.direction) / wos.pdf;
                float z = enteredTop ? thickness : 0;
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
                        //PBRT_DBG("After RR with q = %f, beta: %f %f %f %f\n", q, beta[0],beta[1], beta[2], beta[3]);
                    }

                    // Account for media between layers and possibly scatter
                    if (mediaAlbedo == glm::vec3(0)) {
                        // Advance to next layer boundary and update _beta_ for transmittance
                        z = (z == thickness) ? 0 : thickness;
                        beta *= Transmittance(thickness, w);

                    }
                    else {//<-------------------------------------------------------------CURRENTLY NOT HAPPENING!!!!!
                        // Sample medium scattering for layered BSDF evaluation
                        float sigma_t = 1;
                        //float dz = SampleExponential(r(), sigma_t / std::abs(w.z));
                        float dz = -glm::log(1.0f - r()) / (sigma_t / glm::abs(w.z));
                        float zp = w.z > 0.0f ? (z + dz) : (z - dz);
                        //DCHECK_RARE(1e-5, z == zp);
                        if (z == zp)
                            continue;
                        if (0 < zp && zp < thickness) {
                            // Handle scattering event in layered BSDF medium
                            // Account for scattering through _exitInterface_ using _wis_
                            float wt = 1;
                            if (!exit_isSpecular)//!IsSpecular(exitInterface.Flags())
                                wt = PowerHeuristic(1, wis.pdf, 1, Phase_PDF(-w, -wis.direction, g));
                            f += beta * mediaAlbedo * PhaseFunction_p(-w, -wis.direction, g) * wt *
                                Transmittance(zp - exitZ, wis.direction) * wis.color / wis.pdf;

                            // Sample phase function and update layered path state
                            glm::vec2 u{ r(), r() };
                            //pstd::optional<PhaseFunctionSample> ps = phase.Sample_p(-w, u);
                            PhaseFunctionSample ps = PhaseFunction_Sample_p(-w, u, g);
                            if (ps.pdf == 0 || ps.wi.z == 0)
                                continue;
                            beta *= mediaAlbedo * ps.p / ps.pdf;
                            w = ps.wi;
                            z = zp;

                            // Possibly account for scattering through _exitInterface_
                            if (((z < exitZ && w.z > 0) || (z > exitZ && w.z < 0)) && !exit_isSpecular) {//!IsSpecular(exitInterface.Flags())
                                // Account for scattering through _exitInterface_
                                glm::vec3 fExit = Layer_f(exitInterface_top, surface, -w, wi, mode);
                                if (fExit != glm::vec3(0)) {
                                    //float exitPDF = exitInterface.PDF(-w, wi, mode, BxDFReflTransFlags::Transmission);
                                    float exitPDF = Layer_PDF(exitInterface_top, surface, -w, wi, false, true);
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
                        //pstd::optional<BSDFSample> bs = exitInterface.Sample_f(-w, uc, Point2f(r(), r()), mode, BxDFReflTransFlags::Reflection);
                        BSDFSample bs;
                        bool bs_test = Layer_Sample_f(exitInterface_top, randomSeed, surface, -w, bs, mode, true, false);
                        if (!bs_test || bs.color == glm::vec3(0) || bs.pdf == 0 || bs.direction.z == 0)
                            break;
                        beta *= bs.color * SpherGeom::AbsCosTheta(bs.direction) / bs.pdf;
                        w = bs.direction;
                    }
                    else {
                        // Account for scattering at _nonExitInterface_
                        if (!nonExit_isSpecular) {//!IsSpecular(nonExitInterface.Flags())
                            // Add NEE contribution along presampled _wis_ direction
                            float wt = 1;
                            if (!exit_isSpecular)//!IsSpecular(exitInterface.Flags())
                                //wt = PowerHeuristic(1, wis->pdf, 1,nonExitInterface.PDF(-w, -wis->wi, mode));
                                wt = PowerHeuristic(1, wis.pdf, 1, Layer_PDF(nonExitInterface_top, surface, -w, -wis.direction, true, true));
                            f += beta * Layer_f(nonExitInterface_top, surface, -w, -wis.direction, mode) *
                                SpherGeom::AbsCosTheta(wis.direction) * wt * Transmittance(thickness, wis.direction) * wis.color / 
                                wis.pdf;
                        }
                        // Sample new direction using BSDF at _nonExitInterface_
                        //float uc = r();
                        //Point2f u(r(), r());
                        //pstd::optional<BSDFSample> bs = nonExitInterface.Sample_f(-w, uc, u, mode, BxDFReflTransFlags::Reflection);
                        BSDFSample bs;
                        bool bs_test = Layer_Sample_f(nonExitInterface_top, randomSeed, surface, -w, bs, mode, true, false);
                        if (!bs_test || bs.color == glm::vec3(0) || bs.pdf == 0 || bs.direction.z == 0)
                            break;
                        beta *= bs.color * SpherGeom::AbsCosTheta(bs.direction) / bs.pdf;
                        w = bs.direction;

                        if (!exit_isSpecular) {//!IsSpecular(exitInterface.Flags())
                            // Add NEE contribution along direction from BSDF sample
                            glm::vec3 fExit = Layer_f(exitInterface_top, surface, -w, wi, mode);
                            if (fExit != glm::vec3(0)) {
                                float wt = 1;
                                if (!nonExit_isSpecular) {//!IsSpecular(nonExitInterface.Flags())
                                    //float exitPDF = exitInterface.PDF(-w, wi, mode, BxDFReflTransFlags::Transmission);
                                    float exitPDF = Layer_PDF(exitInterface_top, surface, -w, wi, false, true);
                                    wt = PowerHeuristic(1, bs.pdf, 1, exitPDF);
                                }
                                f += beta * Transmittance(thickness, bs.direction) * fExit * wt;
                            }
                        }
                    }
                }
            }

            return f / glm::vec3(nSamples);
        }




        __device__ __host__ bool Sample_f(unsigned int& randomSeed, Surface& surface, BSDFSample& sample) {
            Dielectric::TransportMode mode = Dielectric::TransportMode::Radiance;

            const float thickness = 0.01f;
            const int maxDepth = 10;

            const glm::vec3 mediaAlbedo = glm::vec3(0);//Color of the participating medium between the two layers

            //scattering of light (0 = isotropic)
            const float g = 0;

            bool twoSided = true;
            glm::vec3 wo = surface.outgoingRay;

            //CHECK(sampleFlags == BxDFReflTransFlags::All);  // for now
            // Set _wo_ for layered BSDF sampling
            bool flipWi = false;
            if (twoSided && wo.z < 0) {
                wo = -wo;
                flipWi = true;
            }
        
            // Sample BSDF at entrance interface to get initial direction _w_
            bool enteredTop = twoSided || wo.z > 0;
            BSDFSample bs;
            //bool bs_test = enteredTop ? Layer_Sample_f(true, randomSeed, surface, wo, bs, mode, true, true) : Layer_Sample_f(false, randomSeed, surface, wo, bs, mode, true, true);
            bool bs_test = Layer_Sample_f(enteredTop, randomSeed, surface, wo, bs, mode, true, true);
            if (!bs_test || bs.color == glm::vec3(0) || bs.pdf == 0 || bs.direction.z == 0)
                return false;
            if (bs.reflection) {
                if (flipWi)
                    bs.direction = -bs.direction;
                //bs->pdfIsProportional = true;
                sample = bs;
                return true;
            }
            glm::vec3 w = bs.direction;
            bool specularPath = bs.specular;
        
            // Declare _RNG_ for layered BSDF sampling
            /*RNG rng(Hash(GetOptions().seed, wo), Hash(uc, u));
            auto r = [&rng]() {
                return std::min<float>(rng.Uniform<float>(), OneMinusEpsilon);
                };*/

            unsigned int newSeed = RandomOptix::tea<16>(wo.x * 1000, wo.y * 1000);
            newSeed = RandomOptix::tea<16>(newSeed, randomSeed);

            auto r = [&newSeed]() {
                return RandomOptix::rnd(newSeed);
                };
        
            // Declare common variables for layered BSDF sampling
            glm::vec3 f = bs.color * SpherGeom::AbsCosTheta(bs.direction);
            float pdf = bs.pdf;
            float z = enteredTop ? thickness : 0;
            //HGPhaseFunction phase(g);
        
            for (int depth = 0; depth < maxDepth; ++depth) {
                // Follow random walk through layers to sample layered BSDF
                // Possibly terminate layered BSDF sampling with Russian Roulette
                float rrBeta = SaveMax(f) / pdf;
                if (depth > 3 && rrBeta < 0.25f) {
                    float q = glm::max(0.0f, 1.0f - rrBeta);
                    if (r() < q)
                        return false;
                    pdf *= 1.0f - q;
                }
                if (w.z == 0)
                    return false;
        
                if (mediaAlbedo != glm::vec3(0)) {
                    // Sample potential scattering event in layered medium
                    float sigma_t = 1;
                    //float dz = SampleExponential(r(), sigma_t / AbsCosTheta(w));
                    float dz = -glm::log(1.0f - r()) / (sigma_t / SpherGeom::AbsCosTheta(w));
                    float zp = w.z > 0 ? (z + dz) : (z - dz);
                    //CHECK_RARE(1e-5, zp == z);
                    if (zp == z)
                        return false;
                    if (0 < zp && zp < thickness) {
                        // Update path state for valid scattering event between interfaces
                        //pstd::optional<PhaseFunctionSample> ps = phase.Sample_p(-w, Point2f(r(), r()));
                        PhaseFunctionSample ps = PhaseFunction_Sample_p(-w, glm::vec2(r(), r()), g);
                        if (ps.pdf == 0 || ps.wi.z == 0)
                            return false;
                        f *= mediaAlbedo * ps.p;
                        pdf *= ps.pdf;
                        specularPath = false;
                        w = ps.wi;
                        z = zp;
        
                        continue;
                    }
                    z = glm::clamp(zp, 0.0f, thickness);
                    /*if (z == 0)
                        DCHECK_LT(w.z, 0);
                    else
                        DCHECK_GT(w.z, 0);*/
        
                }
                else {
                    // Advance to the other layer interface
                    z = (z == thickness) ? 0 : thickness;
                    f *= Transmittance(thickness, w);
                }
                // Initialize _interface_ for current interface surface
                //TopOrBottomBxDF<TopBxDF, BottomBxDF> interface;
                bool interface_top;
                if (z == 0)
                    interface_top = false;//bottom
                else
                    interface_top = true;//top
        
                // Sample interface BSDF to determine new path direction
                //float uc = r();
                //Point2f u(r(), r());
                BSDFSample bs;
                //pstd::optional<BSDFSample> bs = interface.Sample_f(-w, uc, u, mode);
                bool bs_test = Layer_Sample_f(interface_top, randomSeed, surface, -w, bs, mode, true, true);
                if (!bs_test || bs.color == glm::vec3(0) || bs.pdf == 0 || bs.direction.z == 0)
                    return false;
                f *= bs.color;
                pdf *= bs.pdf;
                specularPath &= bs.specular;
                w = bs.direction;
        
                // Return _BSDFSample_ if path has left the layers
                if (bs.transmission) {
                    //BxDFFlags flags = SameHemisphere(wo, w) ? BxDFFlags::Reflection : BxDFFlags::Transmission;
                    //flags |= specularPath ? BxDFFlags::Specular : BxDFFlags::Glossy;
                    if (flipWi)
                        w = -w;
                    //return BSDFSample(f, w, pdf, flags, 1.f, true);

                    sample.color = f;
                    sample.direction = w;
                    sample.pdf = pdf;

                    sample.reflection = SpherGeom::SameHemisphere(wo, w);
                    sample.transmission = !sample.reflection;

                    sample.specular = specularPath;
                    sample.glossy = !sample.specular;

                    return true;
                }
        
                // Scale _f_ by cosine term after scattering at the interface
                f *= SpherGeom::AbsCosTheta(bs.direction);
            }
            return false;
        }
	}
}