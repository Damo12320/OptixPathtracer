#pragma once

#include "glmCUDA.h"

struct RadianceRayData {
    glm::vec3 radiance;
    glm::vec3 beta;

    glm::vec3 nextOrigin;
    glm::vec3 nextDirection;

    int bounceCounter;
    int maxBounces;

    unsigned int randomSeed;

    bool isDebugRay;
};

#pragma region PtrPacking

static __forceinline__ __device__
void* unpackPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__
void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

#pragma endregion


#pragma region PerRayDataGetters

template<typename T>
static __forceinline__ __device__ T* getPRD_1()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

//template<typename T>
//static __forceinline__ __device__ T* getPRD_2()
//{
//    const uint32_t u0 = optixGetPayload_2();
//    const uint32_t u1 = optixGetPayload_3();
//    return reinterpret_cast<T*>(unpackPointer(u0, u1));
//}

//template<typename T>
//static __forceinline__ __device__ T* getPRD_3()
//{
//    const uint32_t u0 = optixGetPayload_4();
//    const uint32_t u1 = optixGetPayload_5();
//    return reinterpret_cast<T*>(unpackPointer(u0, u1));
//}

#pragma endregion

__device__ bool IsDebugRay() {
    RadianceRayData* rayData = (RadianceRayData*)getPRD_1<RadianceRayData>();
    return rayData->isDebugRay;
}