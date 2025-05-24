#pragma once

#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "../3rdParty/glm/glm.hpp"

#include <memory>


class OpenGLWindow {
public:
	glm::vec2 lastCursorPosition;
public:
	OpenGLWindow(glm::ivec2 viewSize);
	~OpenGLWindow();

	GLFWwindow* window;
};