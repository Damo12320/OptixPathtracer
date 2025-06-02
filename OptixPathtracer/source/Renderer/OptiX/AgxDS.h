#pragma once

#include "glmCUDA.h"

// Source: https://www.shadertoy.com/view/Dt3XDr

__device__ glm::vec3 xyYToXYZ(glm::vec3 xyY)
{
    float Y = xyY.z;
    float X = (xyY.x * Y) / xyY.y;
    float Z = ((1.0f - xyY.x - xyY.y) * Y) / xyY.y;

    return glm::vec3(X, Y, Z);
}

__device__ glm::vec3 Unproject(glm::vec2 xy)
{
    return xyYToXYZ(glm::vec3(xy.x, xy.y, 1));
}

__device__ glm::mat3 PrimariesToMatrix(glm::vec2 xy_red, glm::vec2 xy_green, glm::vec2 xy_blue, glm::vec2 xy_white)
{
    glm::vec3 XYZ_red = Unproject(xy_red);
    glm::vec3 XYZ_green = Unproject(xy_green);
    glm::vec3 XYZ_blue = Unproject(xy_blue);
    glm::vec3 XYZ_white = Unproject(xy_white);

    glm::mat3 temp = glm::mat3(XYZ_red.x, 1.0, XYZ_red.z,
        XYZ_green.x, 1.f, XYZ_green.z,
        XYZ_blue.x, 1.0, XYZ_blue.z);
    glm::vec3 scale = glm::inverse(temp) * XYZ_white;

    return glm::mat3(XYZ_red * scale.x, XYZ_green * scale.y, XYZ_blue * scale.z);
}

__device__ glm::mat3 ComputeCompressionMatrix(glm::vec2 xyR, glm::vec2 xyG, glm::vec2 xyB, glm::vec2 xyW, float compression)
{
    float scale_factor = 1.0 / (1.0 - compression);
    glm::vec2 R = SaveMix(xyW, xyR, scale_factor);
    glm::vec2 G = SaveMix(xyW, xyG, scale_factor);
    glm::vec2 B = SaveMix(xyW, xyB, scale_factor);
    glm::vec2 W = xyW;

    return PrimariesToMatrix(R, G, B, W);
}

__device__ float DualSection(float x, float linear, float peak)
{
    // Length of linear section
    float S = (peak * linear);
    if (x < S) {
        return x;
    }
    else {
        float C = peak / (peak - S);
        return peak - (peak - S) * exp((-C * (x - S)) / peak);
    }
}

__device__ glm::vec3 DualSection(glm::vec3 x, float linear, float peak)
{
    x.x = DualSection(x.x, linear, peak);
    x.y = DualSection(x.y, linear, peak);
    x.z = DualSection(x.z, linear, peak);
    return x;
}

__device__ glm::vec3 AgX_DS(glm::vec3 color_srgb, float exposure, float saturation, float linear, float peak, float compression)
{
    glm::vec3 workingColor = SaveMax(color_srgb, 0.0f) * glm::vec3(pow(2.0, exposure));

    glm::mat3 sRGB_to_XYZ = PrimariesToMatrix(glm::vec2(0.64, 0.33),
        glm::vec2(0.3, 0.6),
        glm::vec2(0.15, 0.06),
        glm::vec2(0.3127, 0.3290));
    glm::mat3 adjusted_to_XYZ = ComputeCompressionMatrix(glm::vec2(0.64, 0.33),
        glm::vec2(0.3, 0.6),
        glm::vec2(0.15, 0.06),
        glm::vec2(0.3127, 0.3290), compression);

    glm::mat3 XYZ_to_adjusted = inverse(adjusted_to_XYZ);
    glm::mat3 sRGB_to_adjusted = sRGB_to_XYZ * XYZ_to_adjusted;

    workingColor = sRGB_to_adjusted * workingColor;
    workingColor = SaveClamp(DualSection(workingColor, linear, peak), 0.0, 1.0);

    glm::vec3 luminanceWeight = glm::vec3(0.2126729, 0.7151522, 0.0721750);
    glm::vec3 desaturation = glm::vec3(dot(workingColor, luminanceWeight));
    workingColor = SaveMix(desaturation, workingColor, saturation);
    workingColor = SaveClamp(workingColor, 0.0, 1.0);

    workingColor = inverse(sRGB_to_adjusted) * workingColor;

    return workingColor;
}