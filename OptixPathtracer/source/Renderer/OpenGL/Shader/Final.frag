#version 430 core

uniform sampler2D image;

in vec2 uv;

out vec4 color;

vec3 AgX_DS(vec3 color_srgb, float exposure, float saturation, float linear, float peak, float compression);

void main() 
{
    vec3 finalColor = texture(image, uv).rgb;

    finalColor = AgX_DS(finalColor, 0.45, 1.06, 0.18, 1.0, 0.1);//Settings taken from IDKEngine (VoxelConeTracing)
    finalColor = clamp(finalColor, vec3(0.0), vec3(1.0));

	color = vec4(finalColor, 1.0);
}







vec3 xyYToXYZ(vec3 xyY)
{
    float Y = xyY.z;
    float X = (xyY.x * Y) / xyY.y;
    float Z = ((1.0f - xyY.x - xyY.y) * Y) / xyY.y;

    return vec3(X, Y, Z);
}

vec3 Unproject(vec2 xy)
{
    return xyYToXYZ(vec3(xy.x, xy.y, 1));				
}

mat3 PrimariesToMatrix(vec2 xy_red, vec2 xy_green, vec2 xy_blue, vec2 xy_white)
{
    vec3 XYZ_red = Unproject(xy_red);
    vec3 XYZ_green = Unproject(xy_green);
    vec3 XYZ_blue = Unproject(xy_blue);
    vec3 XYZ_white = Unproject(xy_white);

    mat3 temp = mat3(XYZ_red.x,	  1.0, XYZ_red.z,
                    XYZ_green.x, 1.f, XYZ_green.z,
                    XYZ_blue.x,  1.0, XYZ_blue.z);
    vec3 scale = inverse(temp) * XYZ_white;

    return mat3(XYZ_red * scale.x, XYZ_green * scale.y, XYZ_blue * scale.z);
}

mat3 ComputeCompressionMatrix(vec2 xyR, vec2 xyG, vec2 xyB, vec2 xyW, float compression)
{
    float scale_factor = 1.0 / (1.0 - compression);
    vec2 R = mix(xyW, xyR, scale_factor);
    vec2 G = mix(xyW, xyG, scale_factor);
    vec2 B = mix(xyW, xyB, scale_factor);
    vec2 W = xyW;

    return PrimariesToMatrix(R, G, B, W);
}

float DualSection(float x, float linear, float peak)
{
    // Length of linear section
    float S = (peak * linear);
    if (x < S) {
        return x;
    } else {
        float C = peak / (peak - S);
        return peak - (peak - S) * exp((-C * (x - S)) / peak);
    }
}

vec3 DualSection(vec3 x, float linear, float peak)
{
    x.x = DualSection(x.x, linear, peak);
    x.y = DualSection(x.y, linear, peak);
    x.z = DualSection(x.z, linear, peak);
    return x;
}

vec3 AgX_DS(vec3 color_srgb, float exposure, float saturation, float linear, float peak, float compression)
{
    vec3 workingColor = max(color_srgb, 0.0f) * pow(2.0, exposure);

    mat3 sRGB_to_XYZ = PrimariesToMatrix(vec2(0.64, 0.33),
                                        vec2(0.3, 0.6), 
                                        vec2(0.15, 0.06), 
                                        vec2(0.3127, 0.3290));
    mat3 adjusted_to_XYZ = ComputeCompressionMatrix(vec2(0.64,0.33),
                                                    vec2(0.3,0.6), 
                                                    vec2(0.15,0.06), 
                                                    vec2(0.3127, 0.3290), compression);
    mat3 XYZ_to_adjusted = inverse(adjusted_to_XYZ);
    mat3 sRGB_to_adjusted = sRGB_to_XYZ * XYZ_to_adjusted;

    workingColor = sRGB_to_adjusted * workingColor;
    workingColor = clamp(DualSection(workingColor, linear, peak), 0.0, 1.0);

    vec3 luminanceWeight = vec3(0.2126729,  0.7151522,  0.0721750);
    vec3 desaturation = vec3(dot(workingColor, luminanceWeight));
    workingColor = mix(desaturation, workingColor, saturation);
    workingColor = clamp(workingColor, 0.0, 1.0);

    workingColor = inverse(sRGB_to_adjusted) * workingColor;

    return workingColor;
}