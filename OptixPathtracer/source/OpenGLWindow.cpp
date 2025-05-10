#include"OpenGLWindow.h"
#include <iostream>
#include "3rdParty/glad.c"

static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

OpenGLWindow::OpenGLWindow(glm::ivec2 viewSize) {
	glfwSetErrorCallback(glfw_error_callback);

	if (!glfwInit())
	{
		// Initialization failed
		std::cout << "GLFW Initialization failed\n";
		exit(EXIT_FAILURE);
	}

	//Create GLFW Window
	window = glfwCreateWindow(viewSize.x, viewSize.y, "OptiX Pathtracer", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(window);
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	glfwSwapInterval(1);
}

OpenGLWindow::~OpenGLWindow() {
	glfwDestroyWindow(window);
	glfwTerminate();
}