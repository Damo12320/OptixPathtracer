// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

//#define GLM_FORCE_CUDA
//#include "../../3rdParty/glm/glm.hpp"

#include <optix_device.h>
#include <cuda_runtime.h>

#include "LaunchParams.h"
#include "MeshSBTData.h"
#include "OptixHelpers.h"

#include "LightsStruct.h"
#include "LightMethods.h"

#include "RayData.h"

#include "PBRT/Conductor.h"
#include "PBRT/Dielectric.h"
#include "PBRT/LambertDiffuse.h"
#include "PBRT/GlossyDiffuse.h"



/*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
extern "C" __constant__ LaunchParams optixLaunchParams;



//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__shadow()
{
    //only decoration (not needed, but coherent)
}


__device__ glm::vec4 SRGB8ToLinear(glm::vec4 color)
{
    glm::vec4 mixAmount = glm::step(glm::vec4(0.04045), color);

    glm::vec4 a = color / glm::vec4(12.92);

    glm::vec4 nom = color + glm::vec4(0.055);
    glm::vec4 bottom = nom / glm::vec4(1.055);
    glm::vec4 b = SavePow(bottom, glm::vec4(2.4));

    return SaveMix(a, b, mixAmount);
}



__device__ void GetVertices(MeshSBTData& sbtData, Surface& surface) {
    surface.vertices[0] = sbtData.modelMatrix * glm::vec4(sbtData.vertex[surface.index.x], 1);
    surface.vertices[1] = sbtData.modelMatrix * glm::vec4(sbtData.vertex[surface.index.y], 1);
    surface.vertices[2] = sbtData.modelMatrix * glm::vec4(sbtData.vertex[surface.index.z], 1);
}

__device__ void GetNormal(MeshSBTData& sbtData, Surface& surface) {
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------

    glm::vec3 Ng = cross(surface.vertices[1] - surface.vertices[0], surface.vertices[2] - surface.vertices[0]);
    //glm::vec3 Ng = cross(B - A, C - A);

    glm::vec3 Ns = glm::vec3(0);
    if (sbtData.normal) {
        Ns = (1.f - u - v) * sbtData.normal[surface.index.x]
                + u * sbtData.normal[surface.index.y]
                + v * sbtData.normal[surface.index.z];
        Ns = glm::normalize(sbtData.modelMatrix * glm::vec4(Ns, 0.0));
    }

    // ------------------------------------------------------------------
    // face-forward and normalize normals
    // ------------------------------------------------------------------

    if (dot(surface.outgoingRay, Ng) < 0.f) Ng = -Ng;//if not same hemisphere
    Ng = normalize(Ng);

    if (dot(Ng, Ns) < 0.f)
        Ns = -Ns;
        //Ns -= 2.f * dot(Ng, Ns) * Ng;
    Ns = normalize(Ns);

    surface.gNormal = Ng;
    surface.sNormal = Ns;
}

__device__ glm::vec3 GetSurfacePos(glm::vec3* vertices) {
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    glm::vec3 surfPos
        = (1.f - u - v) * vertices[0]
        + u * vertices[1]
        + v * vertices[2];

    return surfPos;
}

__device__ glm::vec2 GetTextureCoord(MeshSBTData& sbtData, glm::ivec3 index) {
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    glm::vec2 tc
        = (1.f - u - v) * sbtData.texcoord[index.x]
        + u * sbtData.texcoord[index.y]
        + v * sbtData.texcoord[index.z];

    return tc;
}

__device__ void SampleTextures(MeshSBTData& sbtData, glm::vec3& texNormal, Surface& surface) {
    // ------------------------------------------------------------------
    // Get Texture parameters
    // ------------------------------------------------------------------
    float x = surface.texCoord.x;
    float y = surface.texCoord.y;

    if (sbtData.hasAlbedoTexture) {
        glm::vec4 fromTexture = OptixHelpers::Vec4(tex2D<float4>(sbtData.albedoTexture, x, y));
        fromTexture = SRGB8ToLinear(fromTexture);
        surface.albedo *= glm::vec3(fromTexture.x, fromTexture.y, fromTexture.z);
    }

    if (sbtData.hasNormalTexture) {
        glm::vec4 fromTexture = OptixHelpers::Vec4(tex2D<float4>(sbtData.normalTexture, x, y));
        texNormal = glm::vec3(fromTexture.x, fromTexture.y, fromTexture.z);
    }

    if (sbtData.hasMetalRoughTexture) {
        glm::vec4 fromTexture = OptixHelpers::Vec4(tex2D<float4>(sbtData.metalRoughTexture, x, y));
        surface.metallic = fromTexture.r;
        surface.roughness = fromTexture.g;
    }
}

__device__ void BuildTangentSpace(glm::vec3 normal, glm::vec3& tangent, glm::vec3& bitangent) {
    glm::vec3 c1 = glm::cross(normal, glm::vec3(0.0, 0.0, 1.0));
    glm::vec3 c2 = glm::cross(normal, glm::vec3(0.0, 1.0, 0.0));

    if (glm::length(c1) > glm::length(c2))
    {
        tangent = c1;
    }
    else
    {
        tangent = c2;
    }

    tangent = glm::normalize(tangent);

    bitangent = cross(tangent, normal);
}

__device__ glm::mat3 GetTBN(MeshSBTData& sbtData, Surface& surface) {
    /*const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    glm::vec3 tangent =
        (1.f - u - v) * sbtData.tangents[surface.index.x]
        + u * sbtData.tangents[surface.index.y]
        + v * sbtData.tangents[surface.index.z];

    glm::vec3 bitangent =
        (1.f - u - v) * sbtData.bitangents[surface.index.x]
        + u * sbtData.bitangents[surface.index.y]
        + v * sbtData.bitangents[surface.index.z];

    glm::vec3 T = glm::normalize(glm::vec3(sbtData.modelMatrix * glm::vec4(tangent, 0.0)));
    glm::vec3 B = glm::normalize(glm::vec3(sbtData.modelMatrix * glm::vec4(bitangent, 0.0)));
    glm::vec3 N = glm::normalize(glm::vec3(glm::vec4(surface.sNormal, 0.0)));*/

    glm::vec3 tangent;
    glm::vec3 bitangent;
    BuildTangentSpace(surface.sNormal, tangent, bitangent);

    glm::vec3 T = tangent;
    glm::vec3 B = bitangent;
    glm::vec3 N = surface.sNormal;
    return glm::mat3(T, B, N);
}

#pragma region RayTracingMethods

__device__ float LightVisibility(glm::vec3 pointLightPos, glm::vec3 surfPos, glm::vec3 normal, glm::vec3& lightDirection) {
    glm::vec3 lightDir = pointLightPos - surfPos;
    lightDirection = glm::normalize(lightDir);

    float lightVisibility = 0.0f;
    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer(&lightVisibility, u0, u1);
    optixTrace(optixLaunchParams.traversable,
        OptixHelpers::Float3(surfPos + 1e-3f * normal),
        OptixHelpers::Float3(glm::normalize(lightDir)),
        0,      // tmin
        glm::length(lightDir),  // tmax
        0.0f,       // rayTime
        OptixVisibilityMask(255),
        // For shadow rays: skip any/closest hit shaders and terminate on first
        // intersection with anything. The miss shader is used to mark if the
        // light was visible.
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        SHADOW_RAY_TYPE,            // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        SHADOW_RAY_TYPE,            // missSBTIndex 
        u0, u1);

    return lightVisibility;
}

__device__ void TraceRadiance(glm::vec3 rayOrigin, glm::vec3 rayDirection, float minDistance, float maxDistance, RadianceRayData* rayData) {
    // the values we store the pixelColor pointer in:
    uint32_t dataPtr0, dataPtr1;
    packPointer(rayData, dataPtr0, dataPtr1);

    optixTrace(optixLaunchParams.traversable,
        OptixHelpers::Float3(rayOrigin),
        OptixHelpers::Float3(rayDirection),
        minDistance,    // tmin
        maxDistance,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,//OPTIX_RAY_FLAG_NONE,OPTIX_RAY_FLAG_DISABLE_ANYHIT
        RADIANCE_RAY_TYPE,            // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        RADIANCE_RAY_TYPE,            // missSBTIndex 
        dataPtr0, dataPtr1);
}

#pragma endregion

#pragma region RandomDirections

__device__ glm::vec2 Hash2(unsigned int& seed)
{
    seed += 1;
    float s1 = static_cast<float>(seed);
    seed += 1;
    float s2 = static_cast<float>(seed);

    float x = sinf(s1) * 43758.5453123f;
    float y = sinf(s2) * 22578.1459123f;

    return glm::vec2(x - floorf(x), y - floorf(y)); // fract
}

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

#pragma endregion

__device__ bool GetNewRayDirection(unsigned int& seed, Surface& surface, BSDFSample& bsdfSample) {
    //bool test = PBRT::Dielectric::Sample_f(seed, surface, surface.outgoingRay, bsdfSample, PBRT::Dielectric::TransportMode::Radiance, true, true);
    //bool test = PBRT::GlossyDiffuse::Sample_f(seed, surface, bsdfSample);
    //bool test = PBRT::LambertDiffuse::Sample_f(seed, surface, surface.outgoingRay, bsdfSample, true);
    //bool test = PBRT::Conductor::Sample_f(seed, surface.albedo, surface.roughness, surface.outgoingRay, bsdfSample);
    //return test;

    if (surface.conductor) {
    
        return PBRT::Conductor::Sample_f(seed, surface.albedo, surface.roughness, surface.outgoingRay, bsdfSample);
    }
    else {
        bool test = PBRT::GlossyDiffuse::Sample_f(seed, surface, bsdfSample);
    
        return test;
    }

    //return true;
}

__device__ glm::vec3 BRDF(unsigned int& seed, Surface& surface, const glm::vec3 incommingRay) {
    //return PBRT::Dielectric::f(surface, surface.outgoingRay, incommingRay, PBRT::Dielectric::TransportMode::Radiance);
    //return PBRT::GlossyDiffuse::f(seed, surface, incommingRay);
    //return PBRT::LambertDiffuse::f(surface, surface.outgoingRay, incommingRay);
    //return PBRT::Conductor::f(surface.albedo, surface.roughness, surface.outgoingRay, incommingRay);

    if (surface.conductor) {
        return PBRT::Conductor::f(surface.albedo, surface.roughness, surface.outgoingRay, incommingRay);
    }
    else {
        //return PBRT::Dielectric::f(surface, incommingRay) * surface.albedo;
        //return PBRT::LambertDiffuse::f(surface.roughness, surface.outgoingRay, incommingRay) * surface.albedo;
        return PBRT::GlossyDiffuse::f(seed, surface, incommingRay);
    }

    //Diffuse
    //return mix(surface.albedo / 3.14159265359f, conductor, surface.metallic);
    //return SaveMix(dielectric, conductor, surface.metallic);
}

extern "C" __global__ void __closesthit__radiance()
{
    const glm::vec3 SHADING_SPACE_NORMAL = glm::vec3(0, 0, 1);

    MeshSBTData& sbtData = *(MeshSBTData*)optixGetSbtDataPointer();

    bool isBackFaceHit = optixIsTriangleBackFaceHit();

    Surface surface;

    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    RadianceRayData* rayData = (RadianceRayData*)getPRD_1<RadianceRayData>();


    //Return if maximum depth is reached
    rayData->bounceCounter++;
    if (rayData->bounceCounter > rayData->maxBounces) {
        rayData->endPath = true;
        return;
    }

    surface.outgoingRay = glm::normalize(-OptixHelpers::Vec3(optixGetWorldRayDirection()));
    //glm::vec3 hitPoint = rayData->nextOrigin + optixGetRayTmax() * surface.incommingRay;

    const int primID = optixGetPrimitiveIndex();
    surface.index = sbtData.index[primID];

    //Vertices
    glm::vec3 vertices[3];
    GetVertices(sbtData, surface);
    //Normal
    GetNormal(sbtData, surface);

    //Normals are always pointing to the "outside" of the object
    if (isBackFaceHit) {
        surface.sNormal *= -1.0f;
        surface.gNormal *= -1.0f;
    }

    //Surface Position
    surface.position = GetSurfacePos(surface.vertices);
    //Texture Coord
    surface.texCoord = GetTextureCoord(sbtData, surface.index);


    // ------------------------------------------------------------------
    // Get Texture parameters
    // ------------------------------------------------------------------
    surface.albedo = sbtData.albedoColor;
    surface.metallic = sbtData.metallic;
    surface.roughness = sbtData.roughness;

    glm::vec3 normalTex = glm::vec3(0);
    SampleTextures(sbtData, normalTex, surface);

    surface.conductor = RandomOptix::rnd(rayData->randomSeed) < surface.metallic;

    //Normalmapping
    if (normalTex != glm::vec3(0)) {
        //Get TBN from mesh shading normal to transform the normalTex to worldspace
        glm::mat3 tbn = GetTBN(sbtData, surface);

        //Normalize the normalTex and convert to worldspace
        surface.sNormal = glm::normalize(tbn * (normalTex * glm::vec3(2.0f) - glm::vec3(1.0f)));
    }

    //Get Final TBN to convert everything to shadingspace -> Normal = (0, 0, 1)
    surface.ShadingToWorld = GetTBN(sbtData, surface);
    surface.WorldToShading = glm::transpose(surface.ShadingToWorld);

    //Transform everything to local Shading Coordinate System
    surface.outgoingRay = surface.WorldToShading * surface.outgoingRay;
    //surface.sNormal = surface.WorldToShading * surface.sNormal;//sanity check
    //if (surface.sNormal != glm::vec3(0, 0, 1)) {
    //    Print("Normal: ", surface.sNormal);
    //}

    /*if (surface.albedo == glm::vec3(1)) {
        surface.roughness = 0.99f;
    }*/

    //------------------------------------------------------------------------------------------

    if (rayData->isDebugRay) {
        //Print("added Radiance", (BRDF(rayData->randomSeed, surface, lightDirection)) * fabsf(glm::dot(lightDirection, surface.sNormal)), rayData->bounceCounter);
        printf("Debug Ray \n");
        Print("Position", surface.position, rayData->bounceCounter);
        Print("SurfaceAlbedo", surface.albedo, rayData->bounceCounter);
        Print("SurfaceNormal", surface.sNormal, rayData->bounceCounter);
        Print("GeometryNormal", surface.gNormal, rayData->bounceCounter);
        Print("Roughness", surface.roughness, rayData->bounceCounter);
        Print("Metallic", surface.metallic, rayData->bounceCounter);
    }


    //rayData->beta = glm::vec3(1);
    //rayData->radiance = surface.albedo;
    //return;

    //Sample direct lighting (pointlight)
    //Get Random Light
    float lightPropability = 0.0f;
    PointLight* pointLight = Lighting::GetRandomPointLight(&optixLaunchParams, rayData->randomSeed, lightPropability);

    if (lightPropability > 0.0f) {
        //Get Visibility
        glm::vec3 lightDirection;
        const float lightVisibility = LightVisibility(pointLight->position, surface.position, surface.gNormal, lightDirection);
        lightDirection = surface.WorldToShading * lightDirection;

        if (lightVisibility > 0.0f) {//&& PBRT::SpherGeom::SameHemisphere(surface.outgoingRay, lightDirection) could be an optimisation for only opache scenes
            glm::vec3 spectrum = BRDF(rayData->randomSeed, surface, lightDirection) * AbsDot(lightDirection, SHADING_SPACE_NORMAL);

            if (spectrum != glm::vec3(0)) {
                rayData->radiance += (rayData->beta * spectrum * Lighting::GetSample(surface.position, pointLight)) / (lightPropability * Lighting::GetPDF(pointLight));// ( sampleWeight * surface * light ) / light propability * PDF
            }

            //if (rayData->isDebugRay) {
            //    printf("lightVisibility > 0.0f \n");
            //}

            //if (rayData->isDebugRay) {
            //    Print("added Radiance", spectrum, rayData->bounceCounter);
            //    //Print("BRDF", BRDF(rayData->randomSeed, surface, lightDirection), rayData->bounceCounter);
            //    //Print("Position", surface.position, rayData->bounceCounter);
            //}
        }
    }

    BSDFSample bsdfSample;
    if (!GetNewRayDirection(rayData->randomSeed, surface, bsdfSample)) {
        rayData->endPath = true;

        if (rayData->isDebugRay) {
            printf("End Path \n");
        }

        return;
    }

    //auto SampleUniformHemisphere = [](glm::vec2 u) {
    //    float z = u[0];
    //    //float r = SafeSqrt(1 - Sqr(z));
    //    float r = glm::sqrt(glm::max(0.0f, 1.0f - Sqr(z)));
    //    float phi = 2 * 3.14159265359f * u[1];
    //    return glm::vec3(r * std::cos(phi), r * std::sin(phi), z);
    //    };

    //bsdfSample.direction = RandomHemisphereDirection(rayData->randomSeed, SHADING_SPACE_NORMAL);
    /*bsdfSample.direction = SampleUniformHemisphere(glm::vec2(RandomOptix::rnd(rayData->randomSeed), RandomOptix::rnd(rayData->randomSeed)));
    if (glm::dot(surface.outgoingRay, SHADING_SPACE_NORMAL) * glm::dot(bsdfSample.direction, SHADING_SPACE_NORMAL) < 0) {
        bsdfSample.direction = -bsdfSample.direction;
    }*/
    //bsdfSample.pdf = 1 / (2 * 3.14159265359f);
    //bsdfSample.color = BRDF(rayData->randomSeed, surface, bsdfSample.direction);

    rayData->beta *= bsdfSample.color * AbsDot(bsdfSample.direction, SHADING_SPACE_NORMAL) / bsdfSample.pdf;

    glm::vec3 positionOffset = 1e-3f * surface.gNormal;
    if (glm::dot(bsdfSample.direction, SHADING_SPACE_NORMAL) < 0) {
        positionOffset = -positionOffset;
    }

    rayData->nextOrigin = surface.position + positionOffset;
    rayData->nextDirection = glm::normalize(surface.ShadingToWorld * bsdfSample.direction);

    //rayData->radiance = (surface.gNormal + 1.0f) / 2.0f;
    //rayData->radiance = (surface.ShadingToWorld * surface.sNormal + 1.0f) / 2.0f;
    //rayData->radiance = glm::vec3(surface.metallic);
}

#pragma region RayAnyhit

__device__ bool AlphaCutout() {
    const MeshSBTData& sbtData
        = *(const MeshSBTData*)optixGetSbtDataPointer();

    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const int   primID = optixGetPrimitiveIndex();
    const glm::ivec3 index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute diffuse material color, including diffuse texture, if
    // available
    // ------------------------------------------------------------------
    if (sbtData.hasAlbedoTexture && sbtData.texcoord) {
        const glm::vec2 tc
            = (1.f - u - v) * sbtData.texcoord[index.x]
            + u * sbtData.texcoord[index.y]
            + v * sbtData.texcoord[index.z];

        glm::vec4 fromTexture = OptixHelpers::Vec4(tex2D<float4>(sbtData.albedoTexture, tc.x, tc.y));
        fromTexture = SRGB8ToLinear(fromTexture);

        return fromTexture.a < 0.9;
    }

    return false;
}

extern "C" __global__ void __anyhit__radiance()
{
    if (AlphaCutout()) {
        optixIgnoreIntersection();
    }
}

extern "C" __global__ void __anyhit__shadow()
{
    if (AlphaCutout()) {
        optixIgnoreIntersection();
    }
}

#pragma endregion


//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

#pragma region RayMiss

extern "C" __global__ void __miss__radiance()
{
    RadianceRayData* rayData = (RadianceRayData*)getPRD_1<RadianceRayData>();

    //rayData->radiance = glm::vec3(0.0f);
    rayData->beta = glm::vec3(0);
    rayData->bounceCounter = 100;
}

extern "C" __global__ void __miss__shadow()
{
    // we didn't hit anything, so the light is visible
    float& prd = *(float*)getPRD_1<float>();
    //prd = glm::vec3(1.f);
    prd = 1.0f;
}

#pragma endregion


//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
#pragma region RayGeneration

__device__ void Camera(glm::ivec2 launchIndex, glm::vec3& origin, glm::vec4& ray_dir) {
    const auto& camera = optixLaunchParams.camera;

    // From https://forums.developer.nvidia.com/t/rebuilding-opengl-camera-and-projection-parameters-in-optix/238362

    // X and Y in the projection plane through which the ray shall be shot
    float x_screen = (static_cast<float>(launchIndex.x) + .5f) / static_cast<float>(optixLaunchParams.frame.size.x);
    float y_screen = (static_cast<float>(launchIndex.y) + .5f) / static_cast<float>(optixLaunchParams.frame.size.y);

    // X and Y in normalized device coordinates
    float x_ndc = x_screen * 2.f - 1.f;
    float y_ndc = y_screen * 2.f - 1.f;

    glm::vec4 homogenious_ndc = glm::vec4(x_ndc, y_ndc, 1.f, 1.f);

    glm::vec4 p_viewspace = camera.inverseProjectionMatrix * homogenious_ndc;

    // Transform into world space but get rid of disturbing w-factor
    glm::vec4 p_worldspace = camera.inverseViewMatrix * glm::vec4(p_viewspace.x, p_viewspace.y, p_viewspace.z, 0.f);

    ray_dir = glm::normalize(p_worldspace);
    origin = camera.position;
}

__device__ glm::vec3 SamplePath(glm::ivec2 launchIndex, glm::vec3 origin, glm::vec3 direction, int maxBounces) {
    RadianceRayData raydata{};
    raydata.radiance = glm::vec3(0.0);
    raydata.beta = glm::vec3(1.0);
    raydata.maxBounces = maxBounces;
    raydata.bounceCounter = 0;
    raydata.randomSeed = RandomOptix::tea<16>(optixLaunchParams.frame.size.x * launchIndex.y + launchIndex.x, optixLaunchParams.frame.id);

    raydata.nextOrigin = origin;
    raydata.nextDirection = direction;
    raydata.endPath = false;

    raydata.isDebugRay = false;

    //glm::ivec2 debugPixel = glm::ivec2(650, 120);
    glm::ivec2 debugPixel = glm::ivec2(760, 1079 - 710);//from tev to build in

    if (launchIndex == debugPixel && optixLaunchParams.frame.id == 10) {
        raydata.isDebugRay = true;
    }

    while (!raydata.endPath && raydata.bounceCounter < raydata.maxBounces && glm::length(raydata.beta) > 0.00001f) {//glm::length(raydata.beta) > 0.00001f // raydata.beta != glm::vec3(0)
    
        /*if (raydata.isDebugRay) {
            Print("Beta", raydata.beta);
        }*/
    
        TraceRadiance(raydata.nextOrigin, raydata.nextDirection, 0.0f, 100.0f, &raydata);
    }

    //TraceRadiance(raydata.nextOrigin, raydata.nextDirection, 0.0f, 100.0f, &raydata);

    /*if (launchIndex.x == debugPixel.x || launchIndex.y == debugPixel.y) {
        return glm::vec3(1, 0, 0);
    }*/


    return raydata.radiance;
    //return glm::vec3(1, 1, 1);
}


extern "C" __global__ void __raygen__renderFrame()
{
    glm::ivec2 launchIndex = glm::ivec2();
    launchIndex.x = optixGetLaunchIndex().x;
    launchIndex.y = optixGetLaunchIndex().y;

    //get the ray from the camera
    glm::vec3 origin = glm::vec3();
    glm::vec4 ray_dir = glm::vec4();
    Camera(launchIndex, origin, ray_dir);

    //RadianceRayData raydata{};
    //raydata.radiance = glm::vec3(0.0);
    //raydata.beta = glm::vec3(1.0);
    //raydata.maxBounces = 4;
    //raydata.bounceCounter = 0;
    //raydata.randomSeed = RandomOptix::tea<16>(optixLaunchParams.frame.size.x * launchIndex.y + launchIndex.x, optixLaunchParams.frame.id);
    //
    ////Trace Rays
    //TraceRadiance(origin, ray_dir, 0.1f, 100.0f, &raydata);

    glm::vec3 pathRadiance = SamplePath(launchIndex, origin, ray_dir, optixLaunchParams.maxBounces);

    //const int r = int(255.99f * pathRadiance.x);
    //const int g = int(255.99f * pathRadiance.y);
    //const int b = int(255.99f * pathRadiance.z);

    /*if (pathRadiance.x < 0 || pathRadiance.y < 0 || pathRadiance.z < 0) {
        printf("djkhskdnajskdajshkdnjksndjkandkjnjdksnjdkanskjd");
    }*/

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    //const uint32_t rgba = 0xff000000
        //| (r << 0) | (g << 8) | (b << 16);

    // and write to frame buffer ...
    const uint32_t fbIndex = launchIndex.x + launchIndex.y * optixLaunchParams.frame.size.x;
    optixLaunchParams.frame.colorBuffer[fbIndex] = pathRadiance;
}

#pragma endregion
