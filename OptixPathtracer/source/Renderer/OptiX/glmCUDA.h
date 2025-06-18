#pragma once

#include <cuda_runtime.h>
#include "../../3rdParty/glm/glm.hpp"

__device__ __host__ glm::vec3 SaveClamp(glm::vec3 value, float min, float max) {
    glm::vec3 temp = value;
    temp.x = glm::clamp(value.x, min, max);
    temp.y = glm::clamp(value.y, min, max);
    temp.z = glm::clamp(value.z, min, max);

    return temp;
}

__device__ __host__ glm::vec3 SaveLog2(glm::vec3 value) {
    glm::vec3 temp = value;
    temp.x = log2f(value.x);
    temp.y = log2f(value.y);
    temp.z = log2f(value.z);

    return temp;
}

__device__ __host__ glm::vec3 SavePow(glm::vec3 value, glm::vec3 pow) {
    glm::vec3 temp = value;
    temp.x = powf(value.x, pow.x);
    temp.y = powf(value.y, pow.y);
    temp.z = powf(value.z, pow.z);

    return temp;
}

__device__ __host__ glm::vec3 SavePow(glm::vec3 value, float pow) {
    glm::vec3 temp = value;
    temp.x = powf(value.x, pow);
    temp.y = powf(value.y, pow);
    temp.z = powf(value.z, pow);

    return temp;
}

__device__ __host__ glm::vec3 SaveMix(glm::vec3 value1, glm::vec3 value2, glm::vec3 mixValue) {
    glm::vec3 temp = value1;
    temp.x = value1.x * (1 - mixValue.x) + value2.x * mixValue.x;
    temp.y = value1.y * (1 - mixValue.y) + value2.y * mixValue.y;
    temp.z = value1.z * (1 - mixValue.z) + value2.z * mixValue.z;

    return temp;
}

__device__ __host__ glm::vec3 SaveMix(glm::vec3 value1, glm::vec3 value2, float mixValue) {
    glm::vec3 temp = value1;
    temp.x = value1.x * (1 - mixValue) + value2.x * mixValue;
    temp.y = value1.y * (1 - mixValue) + value2.y * mixValue;
    temp.z = value1.z * (1 - mixValue) + value2.z * mixValue;

    return temp;
}

__device__ __host__ glm::vec2 SaveMix(glm::vec2 value1, glm::vec2 value2, float mixValue) {
    glm::vec2 temp = value1;
    temp.x = value1.x * (1 - mixValue) + value2.x * mixValue;
    temp.y = value1.y * (1 - mixValue) + value2.y * mixValue;

    return temp;
}

__device__ __host__ glm::vec3 SaveMax(glm::vec3 value1, float value2) {
    glm::vec3 temp = value1;
    temp.x = value1.x > value2 ? value1.x : value2;
    temp.y = value1.y > value2 ? value1.y : value2;
    temp.z = value1.z > value2 ? value1.z : value2;

    return temp;
}

__device__ __host__ float SaveMax(glm::vec3 value) {
    return glm::max(glm::max(value.x, value.y), value.z);
}

__device__ __host__ float Sqr(float value) {
    return value * value;
}

__device__ __host__ glm::vec3 Sqr(glm::vec3 value) {
    return value * value;
}

__device__ __host__ float AbsDot(glm::vec3 value1, glm::vec3 value2) {
    return fabsf(glm::dot(value1, value2));
}

#pragma region DebugMethods

__device__ __host__ __host__ void Print(const char* name, glm::vec3 vec) {
    printf("%s : %f, %f, %f \n", name, vec.r, vec.g, vec.b);
}

__device__ __host__ __host__ void Print(const char* name, glm::vec3 vec, int bounce) {
    printf("Bounce %i: %s : %f, %f, %f \n", bounce, name, vec.r, vec.g, vec.b);
}

__device__ __host__ __host__ void Print(const char* name, float value) {
    printf("%s : %f \n", name, value);
}

__device__ __host__ __host__ void Print(const char* name, float value, int bounce) {
    printf("Bounce %i: %s : %f \n", bounce, name, value);
}

__device__ __host__ __host__ bool isnan(glm::vec3 vec) {
    return isnan(vec.x) || isnan(vec.y) || isnan(vec.z);
}

#pragma endregion