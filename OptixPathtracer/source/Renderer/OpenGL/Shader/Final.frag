#version 430 core

uniform sampler2D image;

in vec2 uv;

out vec4 color;

void main() 
{
	color = texture(image, uv);
}