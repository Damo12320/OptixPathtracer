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

#include "Lights.h"
#include "MinimalAgX.h"
#include "AgxDS.h"

#include "RayData.h"

#include "PBRT/Conductor.h"



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

    if (dot(-surface.outgoingRay, Ng) > 0.f) Ng = -Ng;
    Ng = normalize(Ng);

    if (dot(Ng, Ns) < 0.f)
        Ns -= 2.f * dot(Ng, Ns) * Ng;
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

//__device__ void NormalMapping(MeshSBTData& sbtData, Surface& surface, glm::vec3 normalTex) {
//    if (normalTex == glm::vec3(0)) {
//        BuildTangentSpace(surface.sNormal, surface.stangent, surface.sbitangent);
//        return;
//    }
//
//    //worldspace TBN
//    const float u = optixGetTriangleBarycentrics().x;
//    const float v = optixGetTriangleBarycentrics().y;
//
//
//    glm::vec3 tangent = 
//        (1.f - u - v)   * sbtData.tangents[surface.index.x]
//        + u             * sbtData.tangents[surface.index.y]
//        + v             * sbtData.tangents[surface.index.z];
//
//    glm::vec3 bitangent =
//        (1.f - u - v)   * sbtData.bitangents[surface.index.x]
//        + u             * sbtData.bitangents[surface.index.y]
//        + v             * sbtData.bitangents[surface.index.z];
//
//
//    glm::vec3 T = glm::normalize(glm::vec3(sbtData.modelMatrix * glm::vec4(tangent, 0.0)));
//    glm::vec3 B = glm::normalize(glm::vec3(sbtData.modelMatrix * glm::vec4(bitangent, 0.0)));
//    glm::vec3 N = glm::normalize(glm::vec3(sbtData.modelMatrix * glm::vec4(surface.sNormal, 0.0)));
//    glm::mat3 fragTBN = glm::mat3(T, B, N);
//
//
//    Print("Normal", glm::transpose(fragTBN) * surface.sNormal);
//
//    glm::vec3 normal = normalTex * glm::vec3(2.0) - glm::vec3(1.0);
//    surface.sNormal = glm::normalize(fragTBN * normal);
//    BuildTangentSpace(surface.sNormal, surface.stangent, surface.sbitangent);
//}

#pragma region RayTracingMethods

__device__ float LightVisibility(glm::vec3 pointLightPos, glm::vec3 surfPos, glm::vec3 normal, glm::vec3& lightDirection) {
    glm::vec3 lightDir = pointLightPos - surfPos;
    lightDirection = glm::normalize(lightDir);

    float lightVisibility = 0;
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
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
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

__device__ bool GetNewRayDirection(unsigned int& seed, Surface& surface, glm::vec3& newDirection, float& pdf, glm::vec3& sample) {
    const float pdfRandomHemisphere = 1 / (2 * 3.141592654f);//https://ameye.dev/notes/sampling-the-hemisphere/

    float random = RandomOptix::rnd(seed);

    if (random < surface.metallic) {

        return PBRT::Conductor::Sample_f(seed, surface, newDirection, pdf, sample);
    }
    else {
        pdf = pdfRandomHemisphere;
        sample = surface.albedo / 3.14159265359f;
        newDirection = RandomHemisphereDirection(seed, surface.sNormal);
    }

    return true;
}

__device__ glm::vec3 BRDF(Surface& surface, const glm::vec3 incommingRay) {

    //glm::vec3 dielectric = DielectricBRDF::DielectricBRDF(surface, outgoingRay);
    //glm::vec3 conductor = ConductorBRDF::ConductorBRDF(surface, outgoingRay);

    //return SaveMix(dielectric, conductor, surface.metallic);

    glm::vec3 conductor = PBRT::Conductor::f(surface, incommingRay);

    //Diffuse
    return mix(surface.albedo / 3.14159265359f, conductor, surface.metallic);
}

extern "C" __global__ void __closesthit__radiance()
{
    MeshSBTData& sbtData = *(MeshSBTData*)optixGetSbtDataPointer();

    Surface surface;

    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    RadianceRayData* rayData = (RadianceRayData*)getPRD_1<RadianceRayData>();

    surface.outgoingRay = glm::normalize(-OptixHelpers::Vec3(optixGetWorldRayDirection()));
    //glm::vec3 hitPoint = rayData->nextOrigin + optixGetRayTmax() * surface.incommingRay;

    const int primID = optixGetPrimitiveIndex();
    surface.index = sbtData.index[primID];

    //Vertices
    glm::vec3 vertices[3];
    GetVertices(sbtData, surface);
    //Normal
    GetNormal(sbtData, surface);
    //Surface Position
    surface.position = GetSurfacePos(surface.vertices);
    //Texture Coord
    surface.texCoord = GetTextureCoord(sbtData, surface.index);


    // ------------------------------------------------------------------
    // Get Texture parameters
    // ------------------------------------------------------------------
    surface.albedo = sbtData.albedoColor;
    glm::vec3 normalTex = glm::vec3(0);
    surface.metallic = sbtData.metallic;
    surface.roughness = sbtData.roughness;
    SampleTextures(sbtData, normalTex, surface);


    //Normalmapping
    if (normalTex != glm::vec3(0)) {
        //Get TBN from mesh shading normal to transform the normalTex to worldspace
        glm::mat3 tbn = GetTBN(sbtData, surface);

        //Normalize the normalTex and convert to worldspace
        surface.sNormal = glm::normalize(tbn * (normalTex * glm::vec3(2.0f) - glm::vec3(1.0f)));
    }

    /*if (surface.IsEffectifvelySmooth()) {
        surface.roughness = 0.5f;
    }*/

    //Get Final TBN to convert everything to shadingspace -> Normal = (0, 0, 1)
    surface.ShadingToWorld = GetTBN(sbtData, surface);
    surface.WorldToShading = glm::transpose(surface.ShadingToWorld);
    //surface.WorldToShading = glm::inverse(surface.ShadingToWorld);

    //Transform everything to local Shading Coordinate System
    surface.sNormal = surface.WorldToShading * surface.sNormal;
    surface.outgoingRay = surface.WorldToShading * surface.outgoingRay;

    //------------------------------------------------------------------------------------------

    //Return if maximum depth is reached
    rayData->bounceCounter++;
    if (rayData->bounceCounter > rayData->maxBounces) {
        return;
    }

    //Sample direct lighting (pointlight)
    PointLight pointLight;
    pointLight.position = glm::vec3(0, 2, 0);
    pointLight.color = glm::vec3(100);

    glm::vec3 lightDirection;
    const float lightVisibility = LightVisibility(pointLight.position, surface.position, surface.gNormal, lightDirection);
    lightDirection = surface.WorldToShading * lightDirection;

    /*if (rayData->isDebugRay) {
        Print("Beta", rayData->beta, rayData->bounceCounter);
    }*/

    if (lightVisibility > 0) {
        glm::vec3 spectrum = (BRDF(surface, lightDirection)) * AbsDot(lightDirection, surface.sNormal);

        rayData->radiance += ( rayData->beta * spectrum * pointLight.GetSample(surface.position) ) / (1 * pointLight.GetPDF());// ( sampleWeight * surface * light ) / light propability * PDF

        if (rayData->isDebugRay) {
            Print("added Radiance", (BRDF(surface, lightDirection)) * fabsf(glm::dot(lightDirection, surface.sNormal)), rayData->bounceCounter);
            Print("Position", surface.position, rayData->bounceCounter);
        }
    }

    float pdf;
    glm::vec3 newDirection;
    glm::vec3 sample;
    if (!GetNewRayDirection(rayData->randomSeed, surface, newDirection, pdf, sample)) {
        rayData->beta = glm::vec3(0);

        if (rayData->isDebugRay) {
            printf("End Path");
        }

        return;
    }

    /*if (rayData->isDebugRay) {
        Print("sample", sample, rayData->bounceCounter);
        Print("absDot", AbsDot(newDirection, surface.sNormal), rayData->bounceCounter);
    }*/

    rayData->beta *= sample * AbsDot(newDirection, surface.sNormal) / pdf;

    rayData->nextOrigin = surface.position + 1e-3f * surface.gNormal;
    rayData->nextDirection = surface.ShadingToWorld * newDirection;

    //rayData->radiance = (surface.WorldToShading * surface.gNormal + 1.0f) / 2.0f;
    //rayData->radiance = surface.WorldToShading * surface.gNormal;
    //rayData->radiance = surface.ShadingToWorld * surface.sNormal;
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

    raydata.isDebugRay = false;

    glm::ivec2 debugPixel = glm::ivec2(850, 350);

    if (launchIndex == debugPixel && optixLaunchParams.frame.id == 10) {
        raydata.isDebugRay = true;
    }

    while (raydata.bounceCounter < raydata.maxBounces && glm::length(raydata.beta) > 0.00001) {
    
        /*if (launchIndex == glm::ivec2(1, 1)) {
            PrintVector("Beta", raydata.beta);
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

//ACES Approximation from https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
__device__ glm::vec3 TonemapACES(glm::vec3 vector) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;

    glm::vec3 temp = vector;
    temp.x = glm::clamp((vector.x * (a * vector.x + b)) / (vector.x * (c * vector.x + d) + e), 0.0f, 1.0f);
    temp.y = glm::clamp((vector.y * (a * vector.y + b)) / (vector.y * (c * vector.y + d) + e), 0.0f, 1.0f);
    temp.z = glm::clamp((vector.z * (a * vector.z + b)) / (vector.z * (c * vector.z + d) + e), 0.0f, 1.0f);

    return temp;
}


//From IDKEngine (VoxelConeTracing)
__device__ glm::vec3 LinearToSrgb(glm::vec3 linearRgb)
{
    glm::bvec3 cutoff = glm::lessThan(linearRgb, glm::vec3(0.0031308));
    glm::vec3 higher = glm::vec3(1.055) * SavePow(linearRgb, glm::vec3(1.0 / 2.4)) - glm::vec3(0.055);
    glm::vec3 lower = linearRgb * glm::vec3(12.92);
    glm::vec3 result = SaveMix(higher, lower, cutoff);
    return result;
}

//From IDKEngine (VoxelConeTracing)
__device__ glm::vec3 Dither(glm::vec3 color, glm::ivec2 imgCoord)
{
    // Source: https://github.com/turanszkij/WickedEngine/blob/master/WickedEngine/shaders/globals.hlsli#L824
    const float BayerMatrix8[8][8] =
    {
        { 1.0 / 65.0, 49.0 / 65.0, 13.0 / 65.0, 61.0 / 65.0, 4.0 / 65.0, 52.0 / 65.0, 16.0 / 65.0, 64.0 / 65.0 },
        { 33.0 / 65.0, 17.0 / 65.0, 45.0 / 65.0, 29.0 / 65.0, 36.0 / 65.0, 20.0 / 65.0, 48.0 / 65.0, 32.0 / 65.0 },
        { 9.0 / 65.0, 57.0 / 65.0, 5.0 / 65.0, 53.0 / 65.0, 12.0 / 65.0, 60.0 / 65.0, 8.0 / 65.0, 56.0 / 65.0 },
        { 41.0 / 65.0, 25.0 / 65.0, 37.0 / 65.0, 21.0 / 65.0, 44.0 / 65.0, 28.0 / 65.0, 40.0 / 65.0, 24.0 / 65.0 },
        { 3.0 / 65.0, 51.0 / 65.0, 15.0 / 65.0, 63.0 / 65.0, 2.0 / 65.0, 50.0 / 65.0, 14.0 / 65.0, 62.0 / 65.0 },
        { 35.0 / 65.0, 19.0 / 65.0, 47.0 / 65.0, 31.0 / 65.0, 34.0 / 65.0, 18.0 / 65.0, 46.0 / 65.0, 30.0 / 65.0 },
        { 11.0 / 65.0, 59.0 / 65.0, 7.0 / 65.0, 55.0 / 65.0, 10.0 / 65.0, 58.0 / 65.0, 6.0 / 65.0, 54.0 / 65.0 },
        { 43.0 / 65.0, 27.0 / 65.0, 39.0 / 65.0, 23.0 / 65.0, 42.0 / 65.0, 26.0 / 65.0, 38.0 / 65.0, 22.0 / 65.0 }
    };
    glm::uint len = 8 * 8;

    int x = imgCoord.x % 8;
    int y = imgCoord.y % 8;
    float ditherVal = (BayerMatrix8[x][y] - 0.5) / len;

    glm::vec3 dithered = color + ditherVal;

    return dithered;
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

    glm::vec3 pathRadiance = SamplePath(launchIndex, origin, ray_dir, 3);

    //tonemapping
    //pathRadiance = TonemapACES(pathRadiance);


    pathRadiance = AgX_DS(pathRadiance, 0.45, 1.06, 0.18, 1.0, 0.1);//Settings taken from IDKEngine (VoxelConeTracing)
    pathRadiance = SaveClamp(pathRadiance, 0.0, 1.0);


    //pathRadiance = LinearToSrgb(pathRadiance);
    //pathRadiance = Dither(pathRadiance, launchIndex);

    const int r = int(255.99f * pathRadiance.x);
    const int g = int(255.99f * pathRadiance.y);
    const int b = int(255.99f * pathRadiance.z);

    /*if (pathRadiance.x < 0 || pathRadiance.y < 0 || pathRadiance.z < 0) {
        printf("djkhskdnajskdajshkdnjksndjkandkjnjdksnjdkanskjd");
    }*/

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
        | (r << 0) | (g << 8) | (b << 16);

    // and write to frame buffer ...
    const uint32_t fbIndex = launchIndex.x + launchIndex.y * optixLaunchParams.frame.size.x;
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}

#pragma endregion
