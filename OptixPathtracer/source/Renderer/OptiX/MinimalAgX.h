#pragma once

#include "glmCUDA.h"

//https://iolite-engine.com/blog_posts/minimal_agx_implementation?utm_source=chatgpt.com

// 0: Default, 1: Golden, 2: Punchy
#define AGX_LOOK 0

// Mean error^2: 3.6705141e-06
__device__ glm::vec3 agxDefaultContrastApprox(glm::vec3 x) {
    glm::vec3 x2 = x * x;
    glm::vec3 x4 = x2 * x2;

    return 
        + glm::vec3(15.5) * x4 * x2
        - glm::vec3(40.14) * x4 * x
        + glm::vec3(31.96) * x4
        - glm::vec3(6.868) * x2 * x
        + glm::vec3(0.4298) * x2
        + glm::vec3(0.1191) * x
        - glm::vec3(0.00232);
}

__device__ glm::vec3 agx(glm::vec3 val) {
    const glm::mat3 agx_mat = glm::mat3(
        0.842479062253094, 0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772, 0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104);

    const float min_ev = -12.47393f;
    const float max_ev = 4.026069f;

    // Input transform (inset)
    val = agx_mat * val;

    // Log2 space encoding
    val = SaveClamp(SaveLog2(val), min_ev, max_ev);
    val = (val - min_ev) / (max_ev - min_ev);

    // Apply sigmoid function approximation
    val = agxDefaultContrastApprox(val);

    return val;
}

__device__ glm::vec3 agxEotf(glm::vec3 val) {
    const glm::mat3 agx_mat_inv = glm::mat3(
        1.19687900512017, -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433, 1.15107367264116);

    // Inverse input transform (outset)
    val = agx_mat_inv * val;

    // sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
    // NOTE: We're linearizing the output here. Comment/adjust when
    // *not* using a sRGB render target
    val = SavePow(val, glm::vec3(2.2));

    return val;
}

__device__ glm::vec3 agxLook(glm::vec3 val) {
    // Default
    glm::vec3 offset = glm::vec3(0.0);
    glm::vec3 slope  = glm::vec3(1.0);
    glm::vec3 power  = glm::vec3(1.0);
    float sat = 1.0;

#if AGX_LOOK == 1
    // Golden
    slope = glm::vec3(1.0, 0.9, 0.5);
    power = glm::vec3(0.8);
    sat = 0.8;
#elif AGX_LOOK == 2
    // Punchy
    slope = glm::vec3(1.0);
    power = glm::vec3(1.35, 1.35, 1.35);
    sat = 1.4;
#endif

    // ASC CDL
    val = SavePow(val * slope + offset, power);

    const glm::vec3 lw = glm::vec3(0.2126, 0.7152, 0.0722);
    float luma = glm::dot(val, lw);

    return luma + sat * (val - luma);
}