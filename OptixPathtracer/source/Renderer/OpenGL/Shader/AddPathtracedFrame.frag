#version 430 core

uniform sampler2D frameBufferImage;
uniform sampler2D imageToAdd;

uniform float weight;
uniform bool addContinuously;

in vec2 uv;

out vec4 color;

void main() 
{
	vec4 frameBufferColor = texture(frameBufferImage, uv);
	vec4 addedColor = texture(imageToAdd, uv);

	if (addContinuously)
	{
		color = mix(frameBufferColor, addedColor, weight);
	}else
	{
		color = frameBufferColor + addedColor * weight;
	}

	//color = addedColor;
	//color = addedColor * vec4(1, 0, 0, 1);
	//color = vec4(1, 0, 0, 1);
}